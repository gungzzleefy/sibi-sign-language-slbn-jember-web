"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import NavBar from "../components/NavBar";
import Image from "next/image";
import { ChevronLeft, ChevronRight } from "lucide-react";
import {
  initializeMediaPipe,
  detectPoseLandmarks,
  extractKeypoints,
  drawLandmarks,
  isHandDetected,
  calculateShoulderDistance,
} from "@/lib/mediapipeUtils";
import {
  loadModel,
  predictGesture,
  isModelLoaded,
  disposeModel,
} from "@/lib/modelUtils";
import {
  ACTIONS,
  SEQUENCE_LENGTH,
  STABILITY_FRAMES,
  THRESHOLD,
} from "@/config/modelConfig";

const SPOK_SENTENCES = [
  {
    kalimat: "Saya makan sayuran agar kuat",
    spok: { S: "Saya", P: "Makan", O: "Sayuran", K: "Agar Kuat" },
    words: [
      { word: "Saya", role: "S" },
      { word: "Makan", role: "P" },
      { word: "Sayur", role: "O" },
      { word: "An", role: "O" },
      { word: "Agar", role: "K" },
      { word: "Kuat", role: "K" },
    ],
  },
  {
    kalimat: "Ibu siapkan kue malam",
    spok: { S: "Ibu", P: "Siapkan", O: "Kue", K: "Malam" },
    words: [
      { word: "Ibu", role: "S" },
      { word: "Siap", role: "P" },
      { word: "Kan", role: "P" },
      { word: "Kue", role: "O" },
      { word: "Malam", role: "K" },
    ],
  },
  {
    kalimat: "Kakak makan obat agar kuat",
    spok: { S: "Kakak", P: "Makan", O: "Obat", K: "Agar Kuat" },
    words: [
      { word: "Kakak", role: "S" },
      { word: "Makan", role: "P" },
      { word: "Obat", role: "O" },
      { word: "Agar", role: "K" },
      { word: "Kuat", role: "K" },
    ],
  },
  {
    kalimat: "Saya siapkan gelas untuk ibu",
    spok: { S: "Saya", P: "Siapkan", O: "Gelas", K: "Untuk Ibu" },
    words: [
      { word: "Saya", role: "S" },
      { word: "Siap", role: "P" },
      { word: "Kan", role: "P" },
      { word: "Gelas", role: "O" },
      { word: "Untuk", role: "K" },
      { word: "Ibu", role: "K" },
    ],
  },
  {
    kalimat: "Ibu siramkan sayuran malam",
    spok: { S: "Ibu", P: "Siramkan", O: "Sayuran", K: "Malam" },
    words: [
      { word: "Ibu", role: "S" },
      { word: "Siram", role: "P" },
      { word: "Kan", role: "P" },
      { word: "Sayur", role: "O" },
      { word: "An", role: "O" },
      { word: "Malam", role: "K" },
    ],
  },
  {
    kalimat: "Saya makan buah agar kuat",
    spok: { S: "Saya", P: "Makan", O: "Buah", K: "Agar Kuat" },
    words: [
      { word: "Saya", role: "S" },
      { word: "Makan", role: "P" },
      { word: "Buah", role: "O" },
      { word: "Agar", role: "K" },
      { word: "Kuat", role: "K" },
    ],
  },
  {
    kalimat: "Kakak harus rajin sabar",
    spok: { S: "Kakak", P: "Harus", O: "Rajin", K: "Sabar" },
    words: [
      { word: "Kakak", role: "S" },
      { word: "Harus", role: "P" },
      { word: "Rajin", role: "O" },
      { word: "Sabar", role: "K" },
    ],
  },
];

const ROLE_COLORS: Record<string, string> = {
  S: "bg-blue-100 text-blue-700 border-blue-300 dark:bg-blue-900/40 dark:text-blue-300 dark:border-blue-700",
  P: "bg-purple-100 text-purple-700 border-purple-300 dark:bg-purple-900/40 dark:text-purple-300 dark:border-purple-700",
  O: "bg-orange-100 text-orange-700 border-orange-300 dark:bg-orange-900/40 dark:text-orange-300 dark:border-orange-700",
  K: "bg-teal-100 text-teal-700 border-teal-300 dark:bg-teal-900/40 dark:text-teal-300 dark:border-teal-700",
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isCameraOn, setIsCameraOn] = useState(false);

  const [isMirrored, setIsMirrored] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [distance, setDistance] = useState(0);

  const isMirroredRef = useRef(isMirrored);
  const showLandmarksRef = useRef(showLandmarks);

  useEffect(() => {
    isMirroredRef.current = isMirrored;
  }, [isMirrored]);
  useEffect(() => {
    showLandmarksRef.current = showLandmarks;
  }, [showLandmarks]);

  const [videoPage, setVideoPage] = useState(0);
  const [videosPerPage, setVideosPerPage] = useState(3);

  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState({
    gesture: "Menunggu...",
    confidence: 0,
    allProbabilities: Array(ACTIONS.length).fill(0),
    detected: false,
  });
  const [stats, setStats] = useState({ frames: 0, fps: 0 });

  // SPOK Quiz states
  const [currentSentenceIdx, setCurrentSentenceIdx] = useState(0);
  const [currentWordIdx, setCurrentWordIdx] = useState(0);
  const [wordStatuses, setWordStatuses] = useState<
    ("pending" | "correct" | "wrong")[]
  >(Array(SPOK_SENTENCES[0].words.length).fill("pending"));
  const [toastVisible, setToastVisible] = useState(false);
  const [showSentencePicker, setShowSentencePicker] = useState(false);

  const [wordConfidences, setWordConfidences] = useState<number[]>(
    Array(SPOK_SENTENCES[0].words.length).fill(0)
  );

  // Refs for detection loop
  const sequenceRef = useRef<Float32Array[]>([]);
  const predictionBufferRef = useRef<number[]>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const animationIdRef = useRef<number | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Refs for SPOK quiz state (prevent stale closure in detectFrame)
  const currentSentenceIdxRef = useRef(0);
  const currentWordIdxRef = useRef(0);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Cooldown: setelah tiap deteksi, tunggu 2 detik agar user sempat ganti pose
  const lastDetectionTimeRef = useRef<number>(0);
  const DETECTION_COOLDOWN_MS = 2000;

  useEffect(() => {
    currentSentenceIdxRef.current = currentSentenceIdx;
  }, [currentSentenceIdx]);
  useEffect(() => {
    currentWordIdxRef.current = currentWordIdx;
  }, [currentWordIdx]);

  // Reset video page when sentence changes
  useEffect(() => {
    setVideoPage(0);
  }, [currentSentenceIdx]);

  const triggerWrongToast = useCallback(() => {
    setToastVisible(true);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToastVisible(false), 2000);
  }, []);
  const triggerWrongToastRef = useRef(triggerWrongToast);
  useEffect(() => {
    triggerWrongToastRef.current = triggerWrongToast;
  }, [triggerWrongToast]);

  useEffect(
    () => () => {
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    },
    [],
  );

  useEffect(() => {
    if (!offscreenCanvasRef.current && typeof document !== "undefined") {
      offscreenCanvasRef.current = document.createElement("canvas");
    }

    const initializeModels = async () => {
      try {
        setIsLoadingModels(true);
        setInitError(null);

        const mediapipeReady = await initializeMediaPipe();
        if (!mediapipeReady) throw new Error("Failed to initialize MediaPipe");

        const modelReady = await loadModel();
        if (!modelReady) throw new Error("Failed to load AI model");

        console.log("✅ AI Models ready!");
        setIsLoadingModels(false);
      } catch (error) {
        setInitError(error instanceof Error ? error.message : "Unknown error");
        setIsLoadingModels(false);
      }
    };
    initializeModels();

    return () => {
      disposeModel();
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
    };
  }, []);

  // Dynamic video list based on current sentence words
  const currentSentence = SPOK_SENTENCES[currentSentenceIdx];
  const currentSentenceWordNames = Array.from(
    new Set(currentSentence.words.map((w) => w.word)),
  );
  const currentSentenceWordSet = new Set(currentSentenceWordNames);
  const video_kosakata = currentSentenceWordNames.map((word, i) => ({
    id: i + 1,
    src: `/video/${word}.webm`,
    alt: word,
  }));

  useEffect(() => {
    const update = () => {
      const perPage = window.innerWidth >= 1024 ? 3 : 2;
      setVideosPerPage((prev) => {
        if (prev !== perPage) setVideoPage(0);
        return perPage;
      });
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const totalPages = Math.ceil(video_kosakata.length / videosPerPage);
  const visibleVideos = video_kosakata.slice(
    videoPage * videosPerPage,
    videoPage * videosPerPage + videosPerPage,
  );

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
      });
      setStream(mediaStream);
      if (videoRef.current) videoRef.current.srcObject = mediaStream;
      setIsCameraOn(true);
      sequenceRef.current = [];
      predictionBufferRef.current = [];
    } catch (error) {
      console.error("Error accessing camera:", error);
      alert("Tidak dapat mengakses kamera. Pastikan Anda memberikan izin.");
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
      if (videoRef.current) videoRef.current.srcObject = null;
      setIsCameraOn(false);
      setDistance(0);
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      sequenceRef.current = [];
    }
  };

  useEffect(() => {
    return () => {
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, [stream]);

  // Main Detection Loop
  const detectFrame = useCallback(async () => {
    if (
      !videoRef.current ||
      !canvasRef.current ||
      !isCameraOn ||
      !isModelLoaded() ||
      videoRef.current.videoWidth === 0
    ) {
      animationIdRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    try {
      const now = performance.now();
      const deltaTime = now - lastFrameTimeRef.current;
      lastFrameTimeRef.current = now;

      frameCountRef.current++;
      if (deltaTime > 0) {
        setStats({
          frames: frameCountRef.current,
          fps: Math.round(1000 / deltaTime),
        });
      }

      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;

      let sourceImage: HTMLVideoElement | HTMLCanvasElement = videoRef.current;

      if (isMirroredRef.current && offscreenCanvasRef.current) {
        if (offscreenCanvasRef.current.width !== videoRef.current.videoWidth) {
          offscreenCanvasRef.current.width = videoRef.current.videoWidth;
          offscreenCanvasRef.current.height = videoRef.current.videoHeight;
        }

        const ctx = offscreenCanvasRef.current.getContext("2d");
        if (ctx) {
          ctx.save();
          ctx.translate(offscreenCanvasRef.current.width, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(
            videoRef.current,
            0,
            0,
            offscreenCanvasRef.current.width,
            offscreenCanvasRef.current.height,
          );
          ctx.restore();
          sourceImage = offscreenCanvasRef.current;
        }
      }

      const detectionResult = await detectPoseLandmarks(sourceImage, now);

      if (detectionResult) {
        const dist = calculateShoulderDistance(
          detectionResult,
          canvasRef.current.width,
          canvasRef.current.height,
        );
        setDistance(dist);
        drawLandmarks(
          canvasRef.current,
          detectionResult,
          showLandmarksRef.current,
        );
      } else {
        setDistance(0);
        drawLandmarks(canvasRef.current, null as any, showLandmarksRef.current);
      }

      const handsDetected = detectionResult
        ? isHandDetected(detectionResult)
        : false;

      if (handsDetected && detectionResult) {
        const keypoints = extractKeypoints(detectionResult);
        sequenceRef.current.push(keypoints);
        sequenceRef.current = sequenceRef.current.slice(-SEQUENCE_LENGTH);

        if (sequenceRef.current.length === SEQUENCE_LENGTH) {
          const result = await predictGesture(sequenceRef.current);

          if (result.predictions.length > 0) {
            const gestureIndex = result.predictions[0];
            predictionBufferRef.current.push(gestureIndex);
            predictionBufferRef.current =
              predictionBufferRef.current.slice(-STABILITY_FRAMES);

            if (predictionBufferRef.current.length === STABILITY_FRAMES) {
              const allSame = predictionBufferRef.current.every(
                (p) => p === predictionBufferRef.current[0],
              );

              if (allSame && result.confidence > THRESHOLD) {
                const nowMs = performance.now();
                const inCooldown =
                  nowMs - lastDetectionTimeRef.current < DETECTION_COOLDOWN_MS;

                if (!inCooldown) {
                  const newGesture = ACTIONS[gestureIndex];
                  const sentenceData =
                    SPOK_SENTENCES[currentSentenceIdxRef.current];

                  // KUNCI SOLUSINYA: Simpan index ke variabel statis sebelum diproses
                  const targetIdx = currentWordIdxRef.current;
                  const expectedWord = sentenceData.words[targetIdx]?.word;

                  if (newGesture === expectedWord) {
  // 1. JIKA BENAR: Update hijau, pindah ke kata selanjutnya, & RESET AI
  lastDetectionTimeRef.current = nowMs;
  predictionBufferRef.current = [];
  sequenceRef.current = [];

  // 👇 1. AMBIL NILAI AKURASI (dijadikan persentase bulat) 👇
  const currentConf = Math.round(result.confidence * 100);

  // Gunakan `targetIdx` yang sudah dikunci, BUKAN `currentWordIdxRef.current`
  setWordStatuses((prev) => {
    const updated = [...prev];
    updated[targetIdx] = "correct";
    return updated;
  });

  // 👇 2. SIMPAN AKURASI KE STATE 👇
  setWordConfidences((prev) => {
    const updated = [...prev];
    updated[targetIdx] = currentConf;
    return updated;
  });

  const newWordIdx = targetIdx + 1;

  if (newWordIdx >= sentenceData.words.length) {
    // Kalimat selesai — lanjut ke berikutnya
    const nextIdx =
      (currentSentenceIdxRef.current + 1) %
      SPOK_SENTENCES.length;
    currentSentenceIdxRef.current = nextIdx;
    currentWordIdxRef.current = 0;
    setTimeout(() => {
      setCurrentSentenceIdx(nextIdx);
      setCurrentWordIdx(0);
      setWordStatuses(
        Array(SPOK_SENTENCES[nextIdx].words.length).fill(
          "pending",
        ),
      );
      // 👇 3. RESET AKURASI UNTUK SOAL BARU 👇
      setWordConfidences(
        Array(SPOK_SENTENCES[nextIdx].words.length).fill(0)
      );
    }, 1200);
  } else {
    // Majukan index setelah kita yakin UI sebelumnya di-update dengan index yang benar
    currentWordIdxRef.current = newWordIdx;
    setCurrentWordIdx(newWordIdx);
  }
} else {
  // 2. JIKA SALAH: Bikin UI Merah, TAPI AI tetap jalan terus
  lastDetectionTimeRef.current = nowMs;
  triggerWrongToastRef.current();

  setWordStatuses((prev) => {
    const updated = [...prev];
    // Pastikan menggunakan targetIdx yang dikunci juga
    if (updated[targetIdx] !== "correct") {
      updated[targetIdx] = "wrong";
    }
    return updated;
  });
}
                }
              }
            }

            setPredictions({
              gesture: ACTIONS[gestureIndex],
              confidence: result.confidence,
              allProbabilities: result.allProbabilities,
              detected: true,
            });
          }
        }
      } else {
        sequenceRef.current = [];
        predictionBufferRef.current = [];
        setPredictions((prev) => ({
          ...prev,
          detected: false,
          gesture: "Menunggu tangan...",
        }));
      }
    } catch (error) {
      console.error("Frame processing error:", error);
    }

    animationIdRef.current = requestAnimationFrame(detectFrame);
  }, [isCameraOn]);

  useEffect(() => {
    if (isCameraOn && isModelLoaded()) {
      lastFrameTimeRef.current = performance.now();
      animationIdRef.current = requestAnimationFrame(detectFrame);
    }
    return () => {
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
    };
  }, [isCameraOn, detectFrame]);

  const allCorrect = wordStatuses.every((s) => s === "correct");
  const correctCount = wordStatuses.filter((s) => s === "correct").length;

  const jumpToSentence = (idx: number) => {
    currentSentenceIdxRef.current = idx;
    currentWordIdxRef.current = 0;
    lastDetectionTimeRef.current = 0;
    setCurrentSentenceIdx(idx);
    setCurrentWordIdx(0);
    setWordStatuses(Array(SPOK_SENTENCES[idx].words.length).fill("pending"));
    setWordConfidences(Array(SPOK_SENTENCES[idx].words.length).fill(0));
    setShowSentencePicker(false);
    sequenceRef.current = [];
    predictionBufferRef.current = [];
  };

  return (
    <div className="min-h-screen w-full font-sans bg-linear-to-b from-blue-50 to-white dark:from-gray-950 dark:to-gray-900">
      {/* Toast — top-left, shown on wrong gesture */}
      {toastVisible && (
        <div className="fixed top-4 left-4 z-50 bg-red-500 text-white px-4 py-3 rounded-xl shadow-lg font-bold text-sm flex items-center gap-2">
          <span className="text-base leading-none">✗</span>
          <span>Salah! Isyarat tidak sesuai</span>
        </div>
      )}

      {/* Modal Pilih Soal */}
      {showSentencePicker && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
          onClick={() => setShowSentencePicker(false)}
        >
          <div
            className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-6 w-full max-w-md flex flex-col gap-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h2 className="font-black text-xl text-gray-800 dark:text-gray-100">
                Pilih Soal Kalimat
              </h2>
              <button
                onClick={() => setShowSentencePicker(false)}
                className="w-8 h-8 flex items-center justify-center rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-500 hover:bg-gray-200 dark:hover:bg-gray-600 font-bold text-lg"
              >
                ✕
              </button>
            </div>
            <div className="flex flex-col gap-2">
              {SPOK_SENTENCES.map((s, i) => (
                <button
                  key={i}
                  onClick={() => jumpToSentence(i)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-xl border text-left transition-all active:scale-98 ${
                    i === currentSentenceIdx
                      ? "bg-purple-50 dark:bg-purple-900/30 border-purple-300 dark:border-purple-700"
                      : "bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600 hover:bg-purple-50 dark:hover:bg-purple-900/20 hover:border-purple-200"
                  }`}
                >
                  <span
                    className={`w-7 h-7 rounded-lg shrink-0 flex items-center justify-center text-xs font-black ${
                      i === currentSentenceIdx
                        ? "bg-purple-500 text-white"
                        : "bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-300"
                    }`}
                  >
                    {i + 1}
                  </span>
                  <div className="flex flex-col min-w-0">
                    <span
                      className={`font-bold text-sm truncate ${i === currentSentenceIdx ? "text-purple-700 dark:text-purple-300" : "text-gray-700 dark:text-gray-200"}`}
                    >
                      {s.kalimat}
                    </span>
                    <span className="text-[10px] text-gray-400 font-medium mt-0.5">
                      S: {s.spok.S} &nbsp;|&nbsp; P: {s.spok.P} &nbsp;|&nbsp; O:{" "}
                      {s.spok.O} &nbsp;|&nbsp; K: {s.spok.K}
                    </span>
                  </div>
                  {i === currentSentenceIdx && (
                    <span className="ml-auto text-purple-500 text-base shrink-0">
                      ●
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <NavBar />
      <main className="w-full py-2 px-4 sm:px-6 lg:px-10 pb-10">
        <div className="w-full bg-[#f0f4ff] dark:bg-gray-800/50 rounded-2xl sm:rounded-3xl p-4 sm:p-6 lg:p-8 flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-[#4251AB] dark:bg-blue-400"></span>
            <span className="text-sm font-bold text-[#4251AB] dark:text-blue-400 uppercase tracking-wider">
              Kamera & AI Setup
            </span>
          </div>

          <div className="w-full grid grid-cols-1 lg:grid-cols-[3fr_1fr] gap-4">
            {/* LEFT: Camera + Action Buttons */}
            <div className="flex flex-col gap-4">
              {/* Camera View */}
              <div className="relative w-full rounded-2xl overflow-hidden bg-[#fece60] aspect-video border border-gray-200 dark:border-gray-700 shadow flex items-center justify-center">
                {isLoadingModels && (
                  <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm text-white">
                    <div className="animate-spin mb-4 w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                    <span className="font-bold tracking-wide">
                      Loading AI Models...
                    </span>
                  </div>
                )}

                {initError && (
                  <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-red-900/80 backdrop-blur-sm text-white p-4 text-center">
                    <span className="text-3xl mb-2">❌</span>
                    <span className="font-bold tracking-wide">
                      MediaPipe Init Error
                    </span>
                    <span className="text-sm font-medium mt-1">
                      {initError}
                    </span>
                  </div>
                )}

                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`absolute inset-0 w-full h-full object-cover transition-transform duration-300 ${isMirrored ? "-scale-x-100" : ""}`}
                />

                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 w-full h-full object-cover z-10 pointer-events-none"
                />

                {!isCameraOn && !isLoadingModels && !initError && (
                  <div className="absolute inset-0 flex items-center justify-center bg-[#fece60]">
                    <span className="text-black text-xl sm:text-2xl font-black absolute z-10 left-4 top-4 sm:left-6 sm:top-6 w-1/2 leading-snug drop-shadow">
                      Hidupkan Kamera untuk Testing AI 📷
                    </span>
                    <Image
                      src="/images/looking-camera-young-handsome-male-barber-uniform-showing-timeout-gesture-isolated-white-background.jpg"
                      width={1000}
                      height={1000}
                      alt="Camera Off"
                      className="absolute inset-0 w-full h-full object-cover mix-blend-multiply opacity-80"
                    />
                  </div>
                )}

                {isCameraOn && (
                  <div className="absolute top-4 left-4 z-20 bg-black/60 backdrop-blur text-white px-3 py-2 rounded-xl border border-white/10 shadow-lg text-sm">
                    <div className="font-mono text-xs opacity-70 mb-1">
                      FPS: {stats.fps}
                    </div>
                    <div className="font-bold flex items-center gap-2">
                      {predictions.detected && (
                        <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e]"></span>
                      )}
                      {predictions.gesture} (
                      {(predictions.confidence * 100).toFixed(0)}%)
                    </div>
                  </div>
                )}

                {isCameraOn && (
                  <div className="absolute top-4 right-4 z-20 bg-black/60 backdrop-blur text-white px-4 py-2 rounded-xl border border-white/10 shadow-lg text-sm flex flex-col items-end min-w-35">
                    <div className="font-mono text-xs opacity-70 mb-1">
                      Jarak: {distance > 0 ? `${distance} cm` : "---"}
                    </div>
                    <div
                      className={`font-bold ${distance > 0 ? (distance < 50 || distance > 100 ? "text-red-400" : "text-green-400") : "text-gray-400"}`}
                    >
                      {distance === 0
                        ? "Bahu Tidak Terdeteksi"
                        : distance < 50
                          ? "<< MUNDUR!"
                          : distance > 100
                            ? "MAJU! >>"
                            : "POSISI OK"}
                    </div>
                  </div>
                )}

                <div className="absolute bottom-0 left-0 w-full px-4 py-3 flex flex-wrap justify-center items-end gap-3 z-20">
                  <button
                    onClick={startCamera}
                    disabled={isCameraOn || isLoadingModels || !!initError}
                    className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl text-white text-xs font-bold shadow transition-all min-w-20 ${
                      isCameraOn || isLoadingModels || !!initError
                        ? "bg-white/20 backdrop-blur cursor-not-allowed opacity-50"
                        : "bg-green-500/80 backdrop-blur hover:bg-green-500 active:scale-95"
                    }`}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      fill="white"
                      viewBox="0 0 16 16"
                    >
                      <path
                        fillRule="evenodd"
                        d="M0 5a2 2 0 0 1 2-2h7.5a2 2 0 0 1 1.983 1.738l3.11-1.382A1 1 0 0 1 16 4.269v7.462a1 1 0 0 1-1.406.913l-3.111-1.382A2 2 0 0 1 9.5 13H2a2 2 0 0 1-2-2z"
                      />
                    </svg>
                    <span>Mulai</span>
                  </button>

                  <button
                    onClick={stopCamera}
                    disabled={!isCameraOn}
                    className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl text-white text-xs font-bold shadow transition-all min-w-20 ${
                      !isCameraOn
                        ? "bg-white/20 backdrop-blur cursor-not-allowed opacity-50"
                        : "bg-red-500/80 backdrop-blur hover:bg-red-500 active:scale-95"
                    }`}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      fill="white"
                      viewBox="0 0 16 16"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10.961 12.365a2 2 0 0 0 .522-1.103l3.11 1.382A1 1 0 0 0 16 11.731V4.269a1 1 0 0 0-1.406-.913l-3.111 1.382A2 2 0 0 0 9.5 3H4.272zm-10.114-9A2 2 0 0 0 0 5v6a2 2 0 0 0 2 2h5.728zm9.746 11.925-10-14 .814-.58 10 14z"
                      />
                    </svg>
                    <span>Berhenti</span>
                  </button>

                  <button
                    onClick={() => setIsMirrored(!isMirrored)}
                    disabled={!isCameraOn}
                    className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl text-white text-xs font-bold shadow transition-all min-w-20 ${
                      !isCameraOn
                        ? "bg-white/20 backdrop-blur cursor-not-allowed opacity-50"
                        : "bg-blue-500/80 backdrop-blur hover:bg-blue-500 active:scale-95"
                    }`}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      fill="white"
                      viewBox="0 0 16 16"
                    >
                      <path d="M11.534 7h3.932a.25.25 0 0 1 .192.41l-1.966 2.36a.25.25 0 0 1-.384 0l-1.966-2.36a.25.25 0 0 1 .192-.41m-11 2h3.932a.25.25 0 0 0 .192-.41L2.692 6.23a.25.25 0 0 0-.384 0L.342 8.59A.25.25 0 0 0 .534 9" />
                      <path
                        fillRule="evenodd"
                        d="M8 3c-1.552 0-2.94.707-3.857 1.818a.5.5 0 1 1-.771-.636A6.002 6.002 0 0 1 13.917 7H12.9A5 5 0 0 0 8 3"
                      />
                    </svg>
                    <span>Mirror: {isMirrored ? "ON" : "OFF"}</span>
                  </button>

                  <button
                    onClick={() => setShowLandmarks(!showLandmarks)}
                    disabled={!isCameraOn}
                    className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl text-white text-xs font-bold shadow transition-all min-w-20 ${
                      !isCameraOn
                        ? "bg-white/20 backdrop-blur cursor-not-allowed opacity-50"
                        : showLandmarks
                          ? "bg-purple-500/80 backdrop-blur hover:bg-purple-500 active:scale-95"
                          : "bg-gray-500/80 backdrop-blur hover:bg-gray-500 active:scale-95"
                    }`}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      fill="white"
                      viewBox="0 0 16 16"
                    >
                      <path d="M10.5 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0" />
                      <path d="M0 8s3-5.5 8-5.5S16 8 16 8s-3 5.5-8 5.5S0 8 0 8m8 3.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7" />
                    </svg>
                    <span>Titik: {showLandmarks ? "ON" : "OFF"}</span>
                  </button>
                </div>
              </div>

              {/* Action Buttons & Video Referensi */}
              <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-4 items-start">
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-11 p-4 flex flex-row sm:flex-col justify-center gap-3 h-full">
                  <button
                    onClick={() => {
                      setCurrentWordIdx(0);
                      currentWordIdxRef.current = 0;
                      setWordStatuses(
                        Array(currentSentence.words.length).fill("pending"),
                      );
                      setWordConfidences(Array(currentSentence.words.length).fill(0));
                      sequenceRef.current = [];
                      predictionBufferRef.current = [];
                    }}
                    className="flex flex-col items-center gap-2 min-w-20 px-4 py-3 bg-[#f0f4ff] dark:bg-gray-700 rounded-xl hover:bg-[#e0e7ff] dark:hover:bg-gray-600 transition-colors active:scale-95"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="26"
                      height="26"
                      fill="currentColor"
                      className="text-[#4251AB] dark:text-blue-400"
                      viewBox="0 0 16 16"
                    >
                      <path d="M11.534 7h3.932a.25.25 0 0 1 .192.41l-1.966 2.36a.25.25 0 0 1-.384 0l-1.966-2.36a.25.25 0 0 1 .192-.41m-11 2h3.932a.25.25 0 0 0 .192-.41L2.692 6.23a.25.25 0 0 0-.384 0L.342 8.59A.25.25 0 0 0 .534 9" />
                      <path
                        fillRule="evenodd"
                        d="M8 3c-1.552 0-2.94.707-3.857 1.818a.5.5 0 1 1-.771-.636A6.002 6.002 0 0 1 13.917 7H12.9A5 5 0 0 0 8 3M3.1 9a5.002 5.002 0 0 0 8.757 2.182.5.5 0 1 1 .771.636A6.002 6.002 0 0 1 2.083 9z"
                      />
                    </svg>
                    <span className="text-xs font-bold text-gray-700 dark:text-gray-200 text-center">
                      Ulangi
                      <br />
                      Deteksi
                    </span>
                  </button>

                  <button
                    onClick={() => setShowSentencePicker(true)}
                    className="flex flex-col items-center gap-2 min-w-20 px-4 py-3 bg-[#f0f4ff] dark:bg-gray-700 rounded-xl hover:bg-[#e0e7ff] dark:hover:bg-gray-600 transition-colors active:scale-95"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="26"
                      height="26"
                      fill="currentColor"
                      className="text-[#4251AB] dark:text-blue-400"
                      viewBox="0 0 16 16"
                    >
                      <path d="M5.933.87a2.89 2.89 0 0 1 4.134 0l.622.638.89-.011a2.89 2.89 0 0 1 2.924 2.924l-.01.89.636.622a2.89 2.89 0 0 1 0 4.134l-.637.622.011.89a2.89 2.89 0 0 1-2.924 2.924l-.89-.01-.622.636a2.89 2.89 0 0 1-4.134 0l-.622-.637-.89.011a2.89 2.89 0 0 1-2.924-2.924l.01-.89-.636-.622a2.89 2.89 0 0 1 0-4.134l.637-.622-.011-.89a2.89 2.89 0 0 1 2.924-2.924l.89.01zM7.002 11a1 1 0 1 0 2 0 1 1 0 0 0-2 0m1.602-2.027c.04-.534.198-.815.846-1.26.674-.475 1.05-1.09 1.05-1.986 0-1.325-.92-2.227-2.262-2.227-1.02 0-1.792.492-2.1 1.29A1.7 1.7 0 0 0 6 5.48c0 .393.203.64.545.64.272 0 .455-.147.564-.51.158-.592.525-.915 1.074-.915.61 0 1.03.446 1.03 1.084 0 .563-.208.885-.822 1.325-.619.433-.926.914-.926 1.64v.111c0 .428.208.745.585.745.336 0 .504-.24.554-.627" />
                    </svg>
                    <span className="text-xs font-bold text-gray-700 dark:text-gray-200 text-center">
                      Pilih
                      <br />
                      Soal
                    </span>
                  </button>
                </div>

                {/* Video Referensi — kata-kata dari soal aktif */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-11 p-3 flex flex-col gap-2 h-full">
                  <div className="flex items-center justify-between px-1">
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-blue-400"></span>
                      <span className="text-sm font-bold text-blue-500 uppercase tracking-wider">
                        Video Isyarat — Soal {currentSentenceIdx + 1}
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setVideoPage((p) => Math.max(0, p - 1))}
                        disabled={videoPage === 0}
                        className={`p-1.5 rounded-lg transition-all active:scale-95 ${videoPage === 0 ? "bg-gray-100 dark:bg-gray-700 text-gray-300 dark:text-gray-600" : "bg-[#2b7fff] text-white hover:bg-[#1a6eee]"}`}
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>
                      <span className="text-xs font-bold text-gray-400 px-1">
                        {videoPage + 1}/{Math.max(1, totalPages)}
                      </span>
                      <button
                        onClick={() =>
                          setVideoPage((p) => Math.min(totalPages - 1, p + 1))
                        }
                        disabled={
                          totalPages === 0 || videoPage === totalPages - 1
                        }
                        className={`p-1.5 rounded-lg transition-all active:scale-95 ${totalPages === 0 || videoPage === totalPages - 1 ? "bg-gray-100 dark:bg-gray-700 text-gray-300 dark:text-gray-600" : "bg-[#2b7fff] text-white hover:bg-[#1a6eee]"}`}
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    {visibleVideos.map((video) => (
                      <div
                        key={video.id}
                        className="relative w-full rounded-xl overflow-hidden bg-gray-100 dark:bg-gray-700"
                      >
                        <video
                          src={video.src}
                          controls
                          loop
                          className="w-full aspect-video object-cover"
                          preload="metadata"
                        />
                        <span className="absolute left-1.5 top-1.5 z-10 px-2 py-0.5 bg-white/85 dark:bg-gray-900/85 backdrop-blur-sm rounded-md font-bold text-xs text-gray-800 dark:text-gray-100 shadow-sm">
                          {video.alt}
                        </span>
                      </div>
                    ))}
                  </div>

                  <div className="flex justify-center gap-1.5 pt-1 mt-auto">
                    {Array.from({ length: Math.max(1, totalPages) }).map(
                      (_, i) => (
                        <button
                          key={i}
                          onClick={() => setVideoPage(i)}
                          className={`h-2 rounded-full transition-all ${i === videoPage ? "w-5 bg-[#2b7fff]" : "w-2 bg-gray-300 dark:bg-gray-600 hover:bg-gray-400 dark:hover:bg-gray-500"}`}
                        />
                      ),
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* RIGHT: Live Scan AI + SPOK Quiz */}
            <div className="flex flex-col gap-4">
              {/* Live Scan AI — filtered to current sentence words only */}
              <div className="bg-white dark:bg-gray-800 shadow-11 rounded-2xl p-5 flex flex-col min-h-45">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-[#4251AB] dark:bg-blue-400 animate-pulse"></span>
                    <span className="font-black text-lg text-gray-800 dark:text-gray-100">
                      Live Scan AI
                    </span>
                  </div>
                  {!isCameraOn && (
                    <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-400 rounded-md">
                      Offline
                    </span>
                  )}
                </div>
                <div className="w-full h-px bg-gray-100 dark:bg-gray-700 mb-3"></div>
                <div className="flex flex-col gap-2">
                  {ACTIONS.filter((action) =>
                    currentSentenceWordSet.has(action),
                  ).map((action) => {
                    const actionIdx = ACTIONS.indexOf(action);
                    const prob = predictions.allProbabilities[actionIdx] || 0;
                    const percent = Math.round(prob * 100);
                    return (
                      <div
                        key={action}
                        className="flex items-center justify-between p-3 rounded-xl bg-blue-50/50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800/30 relative overflow-hidden"
                      >
                        <div
                          className="absolute left-0 top-0 bottom-0 bg-blue-100 dark:bg-blue-900/30 z-0 transition-all duration-300 ease-out"
                          style={{ width: `${percent}%` }}
                        />
                        <span className="font-semibold text-gray-700 dark:text-gray-200 z-10">
                          {action}
                        </span>
                        <span
                          className={`px-3 py-1 text-white rounded-full text-sm font-bold shadow-sm z-10 transition-colors ${percent > 70 ? "bg-green-500" : percent > 20 ? "bg-amber-400" : "bg-gray-400 dark:bg-gray-600"}`}
                        >
                          {percent}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* SPOK Quiz Panel */}
              <div className="bg-white dark:bg-gray-800 shadow-11 rounded-2xl p-5 flex flex-col grow">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-purple-500"></span>
                    <span className="font-black text-lg text-gray-800 dark:text-gray-100">
                      Soal SPOK
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    {SPOK_SENTENCES.map((_, i) => (
                      <button
                        key={i}
                        onClick={() => jumpToSentence(i)}
                        className={`w-7 h-7 rounded-lg text-xs font-bold transition-all active:scale-90 ${
                          i === currentSentenceIdx
                            ? "bg-purple-500 text-white shadow"
                            : "bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 hover:bg-purple-100 dark:hover:bg-purple-900/40 hover:text-purple-600"
                        }`}
                      >
                        {i + 1}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="w-full h-px bg-gray-100 dark:bg-gray-700 mb-3"></div>

                <p className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wide mb-1">
                  Kalimat
                </p>
                <p className="font-bold text-base text-gray-800 dark:text-gray-100 mb-4 italic">
                  &ldquo;{currentSentence.kalimat}&rdquo;
                </p>

                {/* SPOK breakdown */}
                <div className="grid grid-cols-2 gap-1.5 mb-4">
                  {(["S", "P", "O", "K"] as const).map((role) => (
                    <div
                      key={role}
                      className={`rounded-lg px-2 py-1.5 border text-center text-xs font-bold ${ROLE_COLORS[role]}`}
                    >
                      <div className="text-[9px] uppercase tracking-widest opacity-60 mb-0.5">
                        {role === "S"
                          ? "Subjek"
                          : role === "P"
                            ? "Predikat"
                            : role === "O"
                              ? "Objek"
                              : "Keterangan"}
                      </div>
                      <div>{currentSentence.spok[role]}</div>
                    </div>
                  ))}
                </div>

                {/* Word-by-word progress */}
                <p className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wide mb-2">
                  Urutan Isyarat
                </p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {currentSentence.words.map((wordObj, i) => {
                    const status = wordStatuses[i];
                    const isCurrent = i === currentWordIdx && !allCorrect;

                    // Logika styling yang memprioritaskan kata yang sedang ditarget (isCurrent)
                    let baseClass =
                      "px-3 py-1.5 rounded-lg border font-bold text-sm transition-all duration-200 ";

                    if (status === "correct") {
                      baseClass +=
                        "bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300 border-green-300 dark:border-green-700";
                    } else if (status === "wrong") {
                      // KUNCI UI: Jika salah DAN masih jadi target, kasih background merah TAPI tetap kasih Ring penanda fokus!
                      baseClass += isCurrent
                        ? "bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 border-red-400 ring-2 ring-red-400 ring-offset-1"
                        : "bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 border-red-300 dark:border-red-700";
                    } else if (isCurrent) {
                      // Target kata saat ini (sedang menunggu diisyaratkan)
                      baseClass +=
                        "bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-400 dark:border-blue-500 ring-2 ring-blue-400 ring-offset-1";
                    } else {
                      // Kata yang belum waktunya diisyaratkan
                      baseClass +=
                        "bg-white dark:bg-gray-800 text-gray-400 dark:text-gray-500 border-gray-200 dark:border-gray-600";
                    }

                    return (
                     <span key={i} className={baseClass}>
      <span className="text-[9px] uppercase opacity-50 mr-0.5">
        {wordObj.role}
      </span>
      {wordObj.word}
      
      {/* 👇 UBAH BAGIAN TANDA CENTANG JADI SEPERTI INI 👇 */}
      {status === "correct" && (
        <>
          <span className="ml-1.5 text-[10px] font-black px-1.5 py-0.5 rounded-md bg-green-200 dark:bg-green-800/60 text-green-800 dark:text-green-200">
            {wordConfidences[i]}%
          </span>
          <span className="ml-1 text-green-600 dark:text-green-400 font-black">✓</span>
        </>
      )}
      
      {status === "wrong" && (
        <span className="ml-1 text-red-500 font-black">✗</span>
      )}
    </span>
                    );
                  })}
                </div>

                {/* Progress bar */}
                <div className="w-full bg-gray-100 dark:bg-gray-700 rounded-full h-2 mb-1">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${(correctCount / currentSentence.words.length) * 100}%`,
                    }}
                  />
                </div>
                <p className="text-xs text-gray-400 text-right mb-2">
                  {correctCount} / {currentSentence.words.length} kata
                </p>

                {allCorrect && (
                  <div className="mt-2 p-3 bg-green-100 dark:bg-green-900/50 text-green-800 dark:text-green-300 rounded-xl text-center font-bold text-sm">
                    ✨ Kalimat Selesai! Lanjut ke soal berikutnya...
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
