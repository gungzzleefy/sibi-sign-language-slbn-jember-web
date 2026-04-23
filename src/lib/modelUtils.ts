import { SEQUENCE_LENGTH, ACTIONS } from '@/config/modelConfig';

let ws: WebSocket | null = null;
let isConnected = false;

/**
 * Promise store for pending prediction requests
 */
let pendingPredicts: Array<(value: any) => void> = [];

export async function loadModel(): Promise<boolean> {
  return new Promise((resolve) => {
    try {
      // Guard: jangan jalankan di SSR (server-side Next.js)
      if (typeof window === 'undefined') {
        console.warn('⚠️ loadModel dipanggil di server-side, skip.');
        resolve(false);
        return;
      }

      // Dynamic hostname: otomatis pakai IP yang sama dengan frontend
      // Jadi kalau akses dari 192.168.1.15:3000, ws juga ke 192.168.1.15:8000
      const hostname = window.location.hostname;
      const wsUrl = `ws://${hostname}:8000/ws/predict`;

      console.log(`🔄 Menghubungkan ke Backend WebSocket: ${wsUrl}`);

      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('✅ WebSocket Terhubung ke Backend AI!');
        isConnected = true;
        resolve(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const resolveFn = pendingPredicts.shift();

          if (!resolveFn) return;

          if (data.error) {
            console.error('API Error:', data.error);
            resolveFn({
              predictions: [],
              confidence: 0,
              gesture: 'API Error',
              allProbabilities: [],
            });
            return;
          }

          if (data.prediction) {
            const bestIndex = ACTIONS.indexOf(data.prediction);

            resolveFn({
              predictions: [bestIndex !== -1 ? bestIndex : 0],
              confidence: data.confidence,
              gesture: data.prediction,
              allProbabilities: data.probabilities || [],
            });
          }
        } catch (err) {
          console.error('Gagal membaca balasan WebSocket:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket Error:', error);
        if (!isConnected) resolve(false);
      };

      ws.onclose = () => {
        console.log('❌ WebSocket Terputus dari Backend');
        isConnected = false;
        ws = null;
        // Jangan resolve(false) di sini kalau sudah pernah resolve(true)
        // karena Promise hanya bisa resolve sekali
      };

    } catch (error) {
      console.error('❌ Gagal Connect WebSocket:', error);
      resolve(false);
    }
  });
}

export async function predictGesture(
  sequence: Float32Array[],
): Promise<{
  predictions: number[];
  confidence: number;
  gesture: string;
  allProbabilities: number[];
}> {
  if (!isConnected || !ws || ws.readyState !== WebSocket.OPEN) {
    return {
      predictions: [],
      confidence: 0,
      gesture: 'Koneksi Terputus',
      allProbabilities: [],
    };
  }

  if (sequence.length < SEQUENCE_LENGTH) {
    return {
      predictions: [],
      confidence: 0,
      gesture: 'Memuat Frame...',
      allProbabilities: [],
    };
  }

  return new Promise((resolve) => {
    pendingPredicts.push(resolve);

    // Ubah format float32 dari mediapipe jadi standard numeric array JS -> JSON ke API backend
    const sequenceData = sequence.map((frame) => Array.from(frame));
    ws?.send(JSON.stringify({ sequence: sequenceData }));

    // Timeout pelindung kalau backend tidak merespons
    setTimeout(() => {
      const idx = pendingPredicts.indexOf(resolve);
      if (idx !== -1) {
        pendingPredicts.splice(idx, 1);
        resolve({
          predictions: [],
          confidence: 0,
          gesture: 'Timeout API',
          allProbabilities: [],
        });
      }
    }, 1000);
  });
}

/**
 * Check if model/websocket sudah terhubung
 */
export function isModelLoaded(): boolean {
  return isConnected && ws !== null && ws.readyState === WebSocket.OPEN;
}

/**
 * Dispose model untuk cleanup (panggil saat komponen unmount)
 */
export function disposeModel(): void {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
  ws = null;
  isConnected = false;
  pendingPredicts = [];
}