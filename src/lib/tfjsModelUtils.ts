import * as tf from '@tensorflow/tfjs';
import { SEQUENCE_LENGTH, ACTIONS, KEYPOINTS_CONFIG } from '@/config/modelConfig';

const TOTAL_FEATURES = KEYPOINTS_CONFIG.TOTAL_FEATURES;

/**
 * Custom AttentionLayer matching the Keras implementation.
 */
class AttentionLayerImpl extends tf.layers.Layer {
  static className = 'AttentionLayer';

  private W!: tf.LayerVariable;
  private b!: tf.LayerVariable;

  override build(inputShape: tf.Shape | tf.Shape[]): void {
    const shape = (Array.isArray(inputShape[0])
      ? inputShape[0]
      : inputShape) as tf.Shape;

    const features  = shape[shape.length - 1] as number;
    const timeSteps = shape[shape.length - 2] as number;

    this.W = this.addWeight('variable',   [features, 1],   'float32');
    this.b = this.addWeight('variable_1', [timeSteps, 1],  'float32');
    super.build(inputShape);
  }

  override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const x = (Array.isArray(inputs) ? inputs[0] : inputs) as tf.Tensor3D;
      const W = this.W.read();
      const b = this.b.read();

      const [, timeSteps, features] = x.shape;

      // 1. Hitung score seperti biasa
      const xFlat     = x.reshape([-1, features!]);
      const scoreFlat = tf.matMul(xFlat, W);
      const score     = tf.add(scoreFlat.reshape([-1, timeSteps!, 1]), b.expandDims(0));

      // 2. FIX TF.js: Reshape agar timeSteps jadi dimensi terakhir (hilangkan ukuran 1 di ujung)
      const score2D = score.reshape([-1, timeSteps!]);
      
      // 3. Terapkan softmax (TF.js otomatis akan menghitung di dimensi terakhir)
      const alpha2D = tf.softmax(score2D);
      
      // 4. Kembalikan bentuknya ke 3D: [batch, timeSteps, 1] agar bisa dikalikan
      const alpha = alpha2D.reshape([-1, timeSteps!, 1]);

      // 5. Kalikan dengan input (x) dan jumlahkan
      return tf.sum(tf.mul(x, alpha), 1);
    });
  }

  override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
    const shape = (Array.isArray(inputShape[0])
      ? inputShape[0]
      : inputShape) as tf.Shape;
    return [shape[0], shape[shape.length - 1] as number];
  }

  override getConfig() {
    return super.getConfig();
  }
}

tf.serialization.registerClass(AttentionLayerImpl);

let tfjsModel: tf.LayersModel | null = null;
let isLoaded = false;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyObj = any;

/** FIX 1: Ubah DTypePolicy Object menjadi String murni */
function patchDtype(cfg: AnyObj): void {
  if (!cfg) return;
  if (typeof cfg.dtype === 'object' && cfg.dtype?.config?.name) {
    cfg.dtype = cfg.dtype.config.name as string;
  }
}

/** FIX 2: Ubah inbound_nodes dari Object kembali menjadi Array 3-Dimensi TF.js */
function patchInboundNodes(layer: AnyObj): void {
  if (!Array.isArray(layer.inbound_nodes)) return;

  layer.inbound_nodes = layer.inbound_nodes.map((node: AnyObj) => {
    if (Array.isArray(node)) return node;
    if (!node?.args) return [];

    const connections: AnyObj[] = node.args
      .filter((a: AnyObj) => a?.class_name === '__keras_tensor__')
      .map((a: AnyObj) => {
        const [layerName, nodeIdx, tensorIdx] = a.config.keras_history as [string, number, number];
        return [layerName, nodeIdx, tensorIdx, {}];
      });

    return connections;
  });
}

/** FIX 3: Hapus Regularizer agar memori browser tidak hang */
function stripRegularizers(cfg: AnyObj): void {
  if (!cfg) return;

  const regKeys = [
    'kernel_regularizer',
    'bias_regularizer',
    'activity_regularizer',
    'recurrent_regularizer',
  ];

  for (const key of regKeys) {
    if (cfg[key]) delete cfg[key];
  }

  if (cfg.layer?.config)          stripRegularizers(cfg.layer.config);
  if (cfg.backward_layer?.config) stripRegularizers(cfg.backward_layer.config);
}

/**
 * Remap satu weight name dari format Keras3 ke format yang diharapkan TF.js.
 *
 * TF.js LayersModel membangun nama variabel dari path hierarki objek layer-nya,
 * bukan dari nama yang tertulis di file .bin.
 *
 * Format Keras3 (.bin):
 *   forward_{inner}/lstm_cell/kernel
 *   backward_{inner}/lstm_cell/kernel
 *   forward_{inner}/lstm_cell/recurrent_kernel
 *   ...
 *
 * Format TF.js (yang diharapkan saat loadLayersModel):
 *   {wrapper}/forward_{inner}/lstm_cell/kernel
 *   {wrapper}/backward_{inner}/lstm_cell/kernel
 *
 * ATAU (tergantung versi converter):
 *   {wrapper}/lstm_cell/kernel        ← forward
 *   {wrapper}/lstm_cell_1/kernel      ← backward
 *
 * Kita coba kedua strategi. Strategi pertama (prefix wrapper saja) dicoba
 * dulu karena sesuai error terakhir "bidirectional/forward_lstm/lstm_cell/kernel"
 * sudah HAMPIR benar — artinya TF.js sebenarnya menerima format itu,
 * tapi nama layer-nya tidak cocok. Kita perlu memastikan nama wrapper
 * di topology dan di weight spec 100% sama.
 */
function remapBidiWeight(
  weightName: string,
  bidiMap: Record<string, string>,
): string {
  const isForward  = weightName.startsWith('forward_');
  const isBackward = weightName.startsWith('backward_');

  if (!isForward && !isBackward) return weightName;

  const direction   = isForward ? 'forward_' : 'backward_';
  const parts       = weightName.split('/');            // ["backward_lstm_1", "lstm_cell", "kernel"]
  const firstSeg    = parts[0];                         // "backward_lstm_1"
  const rest        = parts.slice(1).join('/');         // "lstm_cell/kernel"
  
  // Dapatkan base name murni dengan menghapus prefix (misal: "lstm_1")
  const innerName   = firstSeg.slice(direction.length); 

  const wrapperName = bidiMap[innerName];

  if (!wrapperName) {
    console.warn(`[BidiRemap] Tidak ada wrapper untuk: "${weightName}"`);
    return weightName; // Kembalikan apa adanya jika tidak ketemu
  }

  // Format TF.js: {wrapper}/forward_{inner}/{rest}
  // Contoh: "bidirectional_1/backward_lstm_1/lstm_cell/kernel"
  return `${wrapperName}/${firstSeg}/${rest}`;
}

/** FUNGSI UTAMA: Download & Patch JSON di Memori */
async function loadPatchedModel(): Promise<tf.LayersModel> {
  const [modelRes, weightsRes] = await Promise.all([
    fetch('/models/tfjs/model.json'),
    fetch('/models/tfjs/group1-shard1of1.bin'),
  ]);

  const modelJson   = await modelRes.json() as AnyObj;
  const weightsData = await weightsRes.arrayBuffer();

  const modelConfig = modelJson.modelTopology?.model_config?.config ?? {};
  const layers      = modelConfig.layers ?? [];

  const bidiMap: Record<string, string> = {};

  for (const layer of layers) {
    // ── InputLayer ───────────────────────────────────────────
    if (layer.class_name === 'InputLayer' && layer.config?.batch_shape !== undefined) {
      layer.config.batchInputShape = layer.config.batch_shape;
      delete layer.config.batch_shape;
    }

    // ── Bidirectional: FIX TOPOLOGY ──────────────────────────────
    if (layer.class_name === 'Bidirectional' && layer.config?.layer?.config?.name) {
      const wrapperName = layer.config.name as string;

      // KUNCI PERBAIKAN: Bersihkan innerName dari prefix Keras yang terbawa
      let innerName = layer.config.layer.config.name as string;
      if (innerName.startsWith('forward_')) {
        innerName = innerName.replace('forward_', '');
      }

      // Simpan map murni, misal: { "lstm": "bidirectional", "lstm_1": "bidirectional_1" }
      bidiMap[innerName] = wrapperName;

      // Paksa topology menggunakan nama yang bersih agar TF.js tidak double-prefix
      layer.config.layer.config.name = innerName;

      // Bersihkan juga backward_layer jika ada
      if (layer.config.backward_layer?.config?.name) {
         let bName = layer.config.backward_layer.config.name as string;
         if (bName.startsWith('backward_')) {
            bName = bName.replace('backward_', '');
         }
         layer.config.backward_layer.config.name = bName;
      }
    }

    patchDtype(layer.config);
    patchDtype(layer.config?.layer?.config);
    patchDtype(layer.config?.backward_layer?.config);
    patchInboundNodes(layer);
    stripRegularizers(layer.config);
  }

  for (const key of ['input_layers', 'output_layers'] as const) {
    if (Array.isArray(modelConfig[key]) && !Array.isArray(modelConfig[key][0])) {
      modelConfig[key] = [modelConfig[key]];
    }
  }

  const weightSpecs: AnyObj[] = modelJson.weightsManifest?.[0]?.weights ?? [];
  let attentionCount = 0;

  for (const w of weightSpecs) {
    if (w.name.includes('attention_layer/')) {
      const layerName = w.name.split('/')[0];
      const suffix    = attentionCount === 0 ? '' : `_${attentionCount}`;
      w.name          = `${layerName}/variable${suffix}`;
      attentionCount++;
      continue;
    }

    if (w.name.startsWith('forward_') || w.name.startsWith('backward_')) {
      w.name = remapBidiWeight(w.name, bidiMap);
      continue;
    }
  }

  const artifacts: tf.io.ModelArtifacts = {
    modelTopology : modelJson.modelTopology,
    weightSpecs   : weightSpecs,
    weightData    : weightsData,
    format        : modelJson.generatedBy,
    generatedBy   : modelJson.generatedBy,
    convertedBy   : modelJson.convertedBy,
  };

  try {
    const model = await tf.loadLayersModel(tf.io.fromMemory(artifacts));
    return model;
  } catch (err) {
    console.error('=== [DIAGNOSTIC] LOAD GAGAL ===', err);
    throw err;
  }
}

export async function loadModel(): Promise<boolean> {
  if (typeof window === 'undefined') return false;

  try {
    console.log('🔄 Memuat TF.js Model di browser...');
    await tf.setBackend('webgl');
    await tf.ready();

    tfjsModel = await loadPatchedModel();

    // Warm-up
    const warmup    = tf.zeros([1, SEQUENCE_LENGTH, TOTAL_FEATURES]);
    const warmupOut = tfjsModel.predict(warmup) as tf.Tensor;
    await warmupOut.data();
    warmupOut.dispose();
    warmup.dispose();

    isLoaded = true;
    console.log('✅ TF.js Model siap dan berjalan normal!');
    return true;
  } catch (error) {
    console.error('❌ Gagal memuat TF.js Model:', error);
    isLoaded = false;
    return false;
  }
}

export async function predictGesture(
  sequence: Float32Array[],
): Promise<{
  predictions:      number[];
  confidence:       number;
  gesture:          string;
  allProbabilities: number[];
}> {
  if (!isLoaded || !tfjsModel) {
    return { predictions: [], confidence: 0, gesture: 'Model belum dimuat', allProbabilities: [] };
  }
  if (sequence.length < SEQUENCE_LENGTH) {
    return { predictions: [], confidence: 0, gesture: 'Memuat Frame...', allProbabilities: [] };
  }

  const inputData = new Float32Array(SEQUENCE_LENGTH * TOTAL_FEATURES);
  for (let i = 0; i < SEQUENCE_LENGTH; i++) {
    const frame = sequence[i];
    for (let j = 0; j < TOTAL_FEATURES; j++) {
      inputData[i * TOTAL_FEATURES + j] = frame[j] ?? 0;
    }
  }

  const inputTensor = tf.tensor3d(inputData, [1, SEQUENCE_LENGTH, TOTAL_FEATURES]);
  let probabilities: number[];

  try {
    const output  = tfjsModel.predict(inputTensor) as tf.Tensor;
    probabilities = Array.from(await output.data());
    output.dispose();
  } finally {
    inputTensor.dispose();
  }

  const bestIndex = probabilities.indexOf(Math.max(...probabilities));

  return {
    predictions:      [bestIndex],
    confidence:       probabilities[bestIndex],
    gesture:          ACTIONS[bestIndex] ?? 'Unknown',
    allProbabilities: probabilities,
  };
}

export function isModelLoaded(): boolean {
  return isLoaded && tfjsModel !== null;
}

export function disposeModel(): void {
  if (tfjsModel) {
    tfjsModel.dispose();
    tfjsModel = null;
  }
  isLoaded = false;
}