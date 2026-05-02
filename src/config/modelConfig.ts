/**
 * Model Configuration
 * Sesuai dengan training model: model7class.keras
 */

export const ACTIONS = [
  'Saya', 'Makan', 'Obat', 'Agar', 'Kuat', 'Buah', 'Sayur',
  'Ibu', 'An', 'Sabar', 'Siap', 'Gelas', 'Harus', 'Kakak',
  'Kan', 'Kue', 'Malam', 'Rajin', 'Siram', 'Untuk',
];

export const SEQUENCE_LENGTH = 45;

export const THRESHOLD = 0.80; // 80% confidence minimum

export const STABILITY_FRAMES = 4; // Frames untuk stabilisasi prediksi

export const MODEL_PATH = '/models/tfjs_model/model.json';

// Keypoints configuration
export const KEYPOINTS_CONFIG = {
  POSE_KEYPOINTS: 33,
  POSE_DIMENSIONS: 4, // x, y, z, visibility
  LEFT_HAND_KEYPOINTS: 21,
  RIGHT_HAND_KEYPOINTS: 21,
  HAND_DIMENSIONS: 3, // x, y, z
  TOTAL_FEATURES: 258, // (33*4) + (21*3) + (21*3)
};

// MediaPipe drawing colors
export const COLORS = {
  POSE: {
    point: { r: 80, g: 22, b: 10 },
    line: { r: 80, g: 44, b: 121 },
  },
  LEFT_HAND: {
    point: { r: 121, g: 22, b: 76 },
    line: { r: 121, g: 44, b: 250 },
  },
  RIGHT_HAND: {
    point: { r: 245, g: 117, b: 66 },
    line: { r: 245, g: 66, b: 230 },
  },
};

// UI Colors for predictions
export const PREDICTION_COLORS = {
  high: { r: 0, g: 210, b: 0 }, // Green
  medium: { r: 255, g: 165, b: 0 }, // Orange
  low: { r: 255, g: 0, b: 0 }, // Red
};
