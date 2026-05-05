"use client";
import NavBar from "../../components/NavBar";
import { useState, useMemo } from "react";
import { BookOpen, PlayCircle, Search, X } from "lucide-react";

const CARD_COLORS = [
  { bg: "bg-blue-50 dark:bg-blue-950/30", border: "border-blue-200 dark:border-blue-800", badge: "bg-blue-500", label: "text-blue-700 dark:text-blue-400" },
  { bg: "bg-purple-50 dark:bg-purple-950/30", border: "border-purple-200 dark:border-purple-800", badge: "bg-purple-500", label: "text-purple-700 dark:text-purple-400" },
  { bg: "bg-green-50 dark:bg-green-950/30", border: "border-green-200 dark:border-green-800", badge: "bg-green-500", label: "text-green-700 dark:text-green-400" },
  { bg: "bg-orange-50 dark:bg-orange-950/30", border: "border-orange-200 dark:border-orange-800", badge: "bg-orange-500", label: "text-orange-700 dark:text-orange-400" },
  { bg: "bg-pink-50 dark:bg-pink-950/30", border: "border-pink-200 dark:border-pink-800", badge: "bg-pink-500", label: "text-pink-700 dark:text-pink-400" },
  { bg: "bg-yellow-50 dark:bg-yellow-950/30", border: "border-yellow-200 dark:border-yellow-800", badge: "bg-yellow-500", label: "text-yellow-700 dark:text-yellow-400" },
  { bg: "bg-teal-50 dark:bg-teal-950/30", border: "border-teal-200 dark:border-teal-800", badge: "bg-teal-500", label: "text-teal-700 dark:text-teal-400" },
  { bg: "bg-red-50 dark:bg-red-950/30", border: "border-red-200 dark:border-red-800", badge: "bg-red-500", label: "text-red-700 dark:text-red-400" },
  { bg: "bg-indigo-50 dark:bg-indigo-950/30", border: "border-indigo-200 dark:border-indigo-800", badge: "bg-indigo-500", label: "text-indigo-700 dark:text-indigo-400" },
  { bg: "bg-cyan-50 dark:bg-cyan-950/30", border: "border-cyan-200 dark:border-cyan-800", badge: "bg-cyan-500", label: "text-cyan-700 dark:text-cyan-400" },
];

const VIDEO_KOSAKATA = [
  { id: 1,  src: "/video/Agar.webm",  alt: "Agar" },
  { id: 2,  src: "/video/An.webm",    alt: "An" },
  { id: 3,  src: "/video/Buah.webm",  alt: "Buah" },
  { id: 4,  src: "/video/Gelas.webm", alt: "Gelas" },
  { id: 5,  src: "/video/Harus.webm", alt: "Harus" },
  { id: 6,  src: "/video/Ibu.webm",   alt: "Ibu" },
  { id: 7,  src: "/video/Kakak.webm", alt: "Kakak" },
  { id: 8,  src: "/video/Kan.webm",   alt: "Kan" },
  { id: 9,  src: "/video/Kuat.webm",  alt: "Kuat" },
  { id: 10, src: "/video/Kue.webm",   alt: "Kue" },
  { id: 11, src: "/video/Makan.webm", alt: "Makan" },
  { id: 12, src: "/video/Malam.webm", alt: "Malam" },
  { id: 13, src: "/video/Obat.webm",  alt: "Obat" },
  { id: 14, src: "/video/Rajin.webm", alt: "Rajin" },
  { id: 15, src: "/video/Sabar.webm", alt: "Sabar" },
  { id: 16, src: "/video/Saya.webm",  alt: "Saya" },
  { id: 17, src: "/video/Sayur.webm", alt: "Sayur" },
  { id: 18, src: "/video/Siap.webm",  alt: "Siap" },
  { id: 19, src: "/video/Siram.webm", alt: "Siram" },
  { id: 20, src: "/video/Untuk.webm", alt: "Untuk" },
];

export default function KosakataPage() {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredVideos = useMemo(() => {
    const words = searchQuery
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => w.toLowerCase());

    if (words.length === 0) return VIDEO_KOSAKATA;

    return VIDEO_KOSAKATA.filter((video) =>
      words.some((word) => video.alt.toLowerCase().includes(word))
    );
  }, [searchQuery]);

  const matchedWords = useMemo(() => {
    const words = searchQuery
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => w.toLowerCase());
    return words.filter((word) =>
      VIDEO_KOSAKATA.some((v) => v.alt.toLowerCase().includes(word))
    );
  }, [searchQuery]);

  return (
    <div className="min-h-screen w-full bg-linear-to-b from-blue-50 to-white dark:from-gray-950 dark:to-gray-900 font-sans">
      <NavBar />

      <main className="w-full px-4 sm:px-6 lg:px-10 pb-10">

        {/* Page Header */}
        <div className="w-full flex flex-col items-center text-center py-6 sm:py-8 gap-2">
          <div className="flex items-center gap-3 bg-[#4251AB] text-white px-5 py-2.5 rounded-full shadow-md">
            <BookOpen className="w-5 h-5" />
            <span className="text-base sm:text-lg font-bold tracking-wide">Kosakata Bahasa SIBI</span>
          </div>
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-black text-gray-800 dark:text-gray-100 mt-3">
            Pelajari Kosakata Isyarat
          </h1>
          <p className="text-gray-500 dark:text-gray-400 text-sm sm:text-base max-w-lg">
            Tonton video gerakan isyarat dan pelajari setiap kata dengan mudah!
          </p>
          <p className="text-gray-400 dark:text-gray-500 text-xs">
            {VIDEO_KOSAKATA.length} kosakata tersedia
          </p>
        </div>

        {/* Search Bar */}
        <div className="w-full flex justify-center mb-8">
          <div className="relative w-full sm:w-[80%] md:w-[60%] lg:w-[45%]">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Cari kosakata... contoh: saya makan"
              className="w-full pl-11 pr-10 py-3 rounded-2xl border-2 border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 shadow-sm focus:outline-none focus:border-[#4251AB] dark:focus:border-blue-500 transition-colors text-sm sm:text-base"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                aria-label="Hapus pencarian"
              >
                <X className="w-4 h-4 text-gray-400" />
              </button>
            )}
          </div>
        </div>

        {/* Search result info */}
        {searchQuery.trim() && (
          <div className="w-full flex flex-col items-center mb-6 gap-2">
            {matchedWords.length > 0 ? (
              <div className="flex flex-wrap justify-center gap-2">
                {matchedWords.map((word) => (
                  <span
                    key={word}
                    className="bg-[#4251AB] text-white text-xs font-semibold px-3 py-1 rounded-full"
                  >
                    {word}
                  </span>
                ))}
                <span className="text-gray-500 dark:text-gray-400 text-sm self-center">
                  — {filteredVideos.length} video ditemukan
                </span>
              </div>
            ) : (
              <p className="text-gray-400 dark:text-gray-500 text-sm">
                Kata &quot;{searchQuery}&quot; tidak ditemukan dalam daftar kosakata
              </p>
            )}
          </div>
        )}

        {/* Video Grid */}
        {filteredVideos.length > 0 ? (
          <div className="w-full grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5 sm:gap-6">
            {filteredVideos.map((video, index) => {
              const color = CARD_COLORS[index % CARD_COLORS.length];
              return (
                <div
                  key={video.id}
                  className={`flex flex-col rounded-2xl border-2 ${color.border} ${color.bg} overflow-hidden shadow-md hover:shadow-lg transition-shadow`}
                >
                  <div className={`flex items-center gap-2 px-4 py-3 ${color.badge}`}>
                    <PlayCircle className="w-5 h-5 text-white" />
                    <span className="text-white font-black text-lg tracking-wide">
                      {video.alt}
                    </span>
                  </div>

                  <div className="p-3">
                    <video
                      src={video.src}
                      controls
                      className="w-full rounded-xl aspect-video object-cover"
                      preload="metadata"
                    />
                  </div>

                  <div className="px-4 pb-4 pt-1 text-center">
                    <span className={`text-sm font-semibold ${color.label}`}>
                      Gerakan isyarat untuk kata &quot;{video.alt}&quot;
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="w-full flex flex-col items-center justify-center py-20 gap-4 text-gray-400">
            <BookOpen className="w-16 h-16 opacity-30" />
            <p className="text-lg font-semibold">Kosakata tidak ditemukan</p>
            <p className="text-sm">Coba kata lain atau hapus pencarian</p>
          </div>
        )}
      </main>
    </div>
  );
}
