import { KEYPOINTS_CONFIG } from '@/config/modelConfig';
import type { Results } from '@mediapipe/holistic';
import * as mpHolistic from '@mediapipe/holistic';
import * as mpDrawingUtils from '@mediapipe/drawing_utils';

const { drawConnectors, drawLandmarks: mpDrawLandmarks } = mpDrawingUtils as any;
const { Holistic, POSE_CONNECTIONS, HAND_CONNECTIONS } = mpHolistic as any;

let holistic: InstanceType<typeof mpHolistic.Holistic> | null = null;
let resolveDetection: ((results: Results) => void) | null = null;

export async function initializeMediaPipe(): Promise<boolean> {
  if (typeof window === 'undefined') return false;
  
  try {
    holistic = new Holistic({
      locateFile: (file: string) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1635989137/${file}`;
      }
    });

    holistic?.setOptions({
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
      refineFaceLandmarks: false,
      enableSegmentation: false,
      modelComplexity: 1
    });

    holistic?.onResults((results) => {
      if (resolveDetection) {
        resolveDetection(results);
        resolveDetection = null;
      }
    });

    if (holistic) {
      await holistic.initialize();
    }

    console.log('✅ MediaPipe Holistic initialized successfully');
    return true;
  } catch (error) {
    console.error('❌ Failed to initialize MediaPipe Holistic:', error);
    return false;
  }
}

export async function detectPoseLandmarks(
  video: HTMLVideoElement | HTMLCanvasElement,
  timestamp: number
): Promise<Results | null> {
  if (!holistic) return null;

  try {
    return new Promise((resolve) => {
      resolveDetection = resolve;
      holistic!.send({ image: video });
      
      setTimeout(() => {
        if (resolveDetection === resolve) {
          resolveDetection = null;
          resolve(null);
        }
      }, 1000);
    });
  } catch (error) {
    console.error('Error detecting landmarks:', error);
    return null;
  }
}

export function extractKeypoints(detectionResult: Results): Float32Array {
  const keypoints = new Float32Array(KEYPOINTS_CONFIG.TOTAL_FEATURES);
  let featureIndex = 0;

  try {
    if (detectionResult.poseLandmarks) {
      for (let i = 0; i < KEYPOINTS_CONFIG.POSE_KEYPOINTS; i++) {
        const lm = detectionResult.poseLandmarks[i];
        keypoints[featureIndex++] = lm.x ?? 0;
        keypoints[featureIndex++] = lm.y ?? 0;
        keypoints[featureIndex++] = lm.z ?? 0;
        keypoints[featureIndex++] = lm.visibility ?? 0;
      }
    } else {
      featureIndex += KEYPOINTS_CONFIG.POSE_KEYPOINTS * KEYPOINTS_CONFIG.POSE_DIMENSIONS;
    }

    if (detectionResult.leftHandLandmarks) {
      for (let i = 0; i < KEYPOINTS_CONFIG.LEFT_HAND_KEYPOINTS; i++) {
        const lm = detectionResult.leftHandLandmarks[i];
        keypoints[featureIndex++] = lm.x ?? 0;
        keypoints[featureIndex++] = lm.y ?? 0;
        keypoints[featureIndex++] = lm.z ?? 0;
      }
    } else {
      featureIndex += KEYPOINTS_CONFIG.LEFT_HAND_KEYPOINTS * KEYPOINTS_CONFIG.HAND_DIMENSIONS;
    }

    if (detectionResult.rightHandLandmarks) {
      for (let i = 0; i < KEYPOINTS_CONFIG.RIGHT_HAND_KEYPOINTS; i++) {
        const lm = detectionResult.rightHandLandmarks[i];
        keypoints[featureIndex++] = lm.x ?? 0;
        keypoints[featureIndex++] = lm.y ?? 0;
        keypoints[featureIndex++] = lm.z ?? 0;
      }
    } else {
      featureIndex += KEYPOINTS_CONFIG.RIGHT_HAND_KEYPOINTS * KEYPOINTS_CONFIG.HAND_DIMENSIONS;
    }

    return keypoints;
  } catch (error) {
    console.error('Error extracting keypoints:', error);
    return keypoints;
  }
}

export function drawLandmarks(
  canvas: HTMLCanvasElement,
  detectionResult: Results,
  showLandmarks: boolean
) {
  try {
    const canvasCtx = canvas?.getContext('2d');
    if (!canvasCtx) return;

    // Selalu bersihkan canvas di awal frame
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    // Jika toggle mati, berhenti di sini (canvas bersih)
    if (!showLandmarks) return; 

    canvasCtx.save();

    if (detectionResult.poseLandmarks) {
      drawConnectors(canvasCtx, detectionResult.poseLandmarks, POSE_CONNECTIONS, { color: 'rgb(80,44,121)', lineWidth: 1 });
      mpDrawLandmarks(canvasCtx, detectionResult.poseLandmarks, { color: 'rgb(80,22,10)', radius: 1 });
    }

    if (detectionResult.leftHandLandmarks) {
      drawConnectors(canvasCtx, detectionResult.leftHandLandmarks, HAND_CONNECTIONS, { color: 'rgb(121,44,250)', lineWidth: 2 });
      mpDrawLandmarks(canvasCtx, detectionResult.leftHandLandmarks, { color: 'rgb(121,22,76)', radius: 4 });
    }

    if (detectionResult.rightHandLandmarks) {
      drawConnectors(canvasCtx, detectionResult.rightHandLandmarks, HAND_CONNECTIONS, { color: 'rgb(245,66,230)', lineWidth: 2 });
      mpDrawLandmarks(canvasCtx, detectionResult.rightHandLandmarks, { color: 'rgb(245,117,66)', radius: 4 });
    }

    canvasCtx.restore();
  } catch (error) {
    console.error('Error drawing landmarks:', error);
  }
}

export function isHandDetected(detectionResult: Results): boolean {
  if (!detectionResult) return false;
  return !!detectionResult.leftHandLandmarks || !!detectionResult.rightHandLandmarks;
}

export function isPoseDetected(detectionResult: Results): boolean {
  if (!detectionResult) return false;
  return !!detectionResult.poseLandmarks;
}

/**
 * Fitur Kalkulasi Jarak Berdasarkan Bahu
 * Estimasi focal length standar = 700. (Di python butuh kalibrasi manual, 
 * tapi untuk web, nilai 700-800 biasanya aman untuk webcam laptop).
 */
export function calculateShoulderDistance(
  detectionResult: Results,
  imageWidth: number,
  imageHeight: number,
  focalLength: number = 700 
): number {
  if (!detectionResult || !detectionResult.poseLandmarks) return 0;

  const leftShoulder = detectionResult.poseLandmarks[11];
  const rightShoulder = detectionResult.poseLandmarks[12];

  if (!leftShoulder || !rightShoulder) return 0;

  const x1 = leftShoulder.x * imageWidth;
  const y1 = leftShoulder.y * imageHeight;
  const x2 = rightShoulder.x * imageWidth;
  const y2 = rightShoulder.y * imageHeight;

  const wPixel = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));

  if (wPixel < 10) return 0; 

  const W_BAHU = 40; // Rata-rata lebar bahu 40cm
  return Math.round((W_BAHU * focalLength) / wPixel);
}