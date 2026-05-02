import cv2
import numpy as np
import os
import math
import json
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import keras
from keras.layers import Layer
import uvicorn

# ==================== PENGATURAN AWAL ====================
app = FastAPI(title="SIBI SPOK Realtime Backend")

# Mengizinkan akses frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== KELAS CUSTOM KERAS ====================
@keras.saving.register_keras_serializable()
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        return tf.reduce_mean(alpha * tf.pow(1.0 - pt, gamma) * ce)
    return loss_fn


# ==================== KONFIGURASI ====================
# Sesuaikan 'actions' dari Config Anda
ACTIONS = ['Saya', 'Makan', 'Obat', 'Agar', 'Kuat', 'Buah', 'Sayur']
SEQUENCE_LENGTH = 45

# Path model (.keras) yang berada di public folder NextJS
# Karena backend berjalan di folder /backend, parent dir adalah frontend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "public", "models", "model7class.keras")

# Global variables
model = None

# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def load_model_on_startup():
    global model
    print(f"[INFO] Memuat model dari: {MODEL_PATH}")
    
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'AttentionLayer': AttentionLayer,
                    'loss_fn': focal_loss(),
                }
            )
            print("[INFO] ✅ Model Keras berhasil dimuat!")
        except Exception as e:
            print(f"[ERROR] Gagal memuat model: {e}")
    else:
        print(f"[ERROR] ❌ File model tidak ditemukan di lokasi: {MODEL_PATH}")

# ==================== MAIN ENDPOINTS ====================
@app.get("/")
def home():
    return {"message": "Server FastAPI SIBI Deteksi aktif."}


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Client (NextJS) Terhubung (WebSocket).")
    
    try:
        while True:
            data_str = await websocket.receive_text()
            
            if model is None:
                 await websocket.send_text(json.dumps({"error": "Model Keras gagal di-load di server."}))
                 continue
                 
            try:
                payload = json.loads(data_str)
                # Payload yg diharapkan adalah json object dgn element "sequence": daftar array 258 angka
                sequence = payload.get("sequence", [])
                
                if len(sequence) == SEQUENCE_LENGTH:
                    input_seq = np.expand_dims(np.array(sequence), axis=0) # Shape: (1, 45, 258)
                    has_nan = bool(np.isnan(input_seq).any())
                    mean_val = float(np.mean(input_seq))
                    print(f"[DEBUG] has_nan={has_nan} | mean={mean_val:.4f} | min={np.min(input_seq):.4f} | max={np.max(input_seq):.4f}")
                    res = model.predict(input_seq, verbose=0)[0]
                    
                    best_idx = int(np.argmax(res))
                    confidence = float(res[best_idx])
                    
                    word = ACTIONS[best_idx] if best_idx < len(ACTIONS) else "Unknown"
                    
                    response = {
                        "prediction": word,
                        "confidence": confidence,
                        "probabilities": res.tolist()
                    }
                    await websocket.send_text(json.dumps(response))
                else:
                    await websocket.send_text(json.dumps({
                        "error": f"Urutan tidak lengkap: {len(sequence)}"
                    }))
                    
            except Exception as e:
                print(f"[ERROR] Prediksi: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))
                
    except WebSocketDisconnect:
        print("[INFO] Client terputus.")
    except Exception as e:
        print(f"[ERROR] WebSocket: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)