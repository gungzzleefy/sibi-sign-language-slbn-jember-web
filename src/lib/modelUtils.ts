import { SEQUENCE_LENGTH, ACTIONS } from '@/config/modelConfig';

export async function loadModel(): Promise<boolean> {
  // Karena kita pakai REST API, kita anggap model selalu siap dihubungi.
  // Tidak perlu lagi memelihara koneksi WebSocket yang sering putus!
  return true; 
}

export async function predictGesture(
  sequence: Float32Array[],
): Promise<{
  predictions: number[];
  confidence: number;
  gesture: string;
  allProbabilities: number[];
}> {
  if (sequence.length < SEQUENCE_LENGTH) {
    return {
      predictions: [],
      confidence: 0,
      gesture: 'Memuat Frame...',
      allProbabilities: [],
    };
  }

  try {
    const sequenceData = sequence.map((frame) => Array.from(frame));
    const apiUrl = "/api/predict";

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence: sequenceData })
    });

    // 👇 KUNCI SOLUSINYA: Cek apakah server membalas dengan HTML (Error/Loading Screen)
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      // Intip isi HTML-nya sedikit untuk tahu apa masalah di server
      const textResponse = await response.text();
      console.error("❌ Server tidak membalas JSON. Balasan:", textResponse.substring(0, 150));
      return { 
        predictions: [], 
        confidence: 0, 
        gesture: 'Server AI Loading/Sibuk...', 
        allProbabilities: [] 
      };
    }

    const data = await response.json();

    if (data.error) {
      console.error('API Error:', data.error);
      return { predictions: [], confidence: 0, gesture: 'API Error', allProbabilities: [] };
    }

    const bestIndex = ACTIONS.indexOf(data.prediction);

    return {
      predictions: [bestIndex !== -1 ? bestIndex : 0],
      confidence: data.confidence,
      gesture: data.prediction,
      allProbabilities: data.probabilities || [],
    };

  } catch (error) {
    console.error("Gagal mengirim prediksi:", error);
    return {
      predictions: [],
      confidence: 0,
      gesture: 'Gagal Konek Server',
      allProbabilities: [],
    };
  }
}

export function isModelLoaded(): boolean {
  // Selalu return true karena REST API bersifat stateless
  return true;
}

export function disposeModel(): void {
  // Kosongkan fungsi ini karena tidak ada koneksi WebSocket yang perlu ditutup
}