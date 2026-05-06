import { NextRequest, NextResponse } from 'next/server';

const HF_API_URL = "https://agungkurniawanid-sibi-sign-slbn-jember-recognition-backend.hf.space/predict";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    // Retry maksimal 3x jika HF Space belum siap (masih returning HTML)
    for (let attempt = 1; attempt <= 3; attempt++) {
      const response = await fetch(HF_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      console.log(`[PROXY] Attempt ${attempt} - Status: ${response.status}, Content-Type: ${response.headers.get("content-type")}`);

      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        const data = await response.json();
        return NextResponse.json(data);
      }

      // Kalau HTML, tunggu sebentar lalu retry
      const text = await response.text();
      console.error(`[PROXY] Attempt ${attempt} - HF Space balas HTML (status ${response.status}):`, text.substring(0, 300));

      if (attempt < 3) {
        await new Promise(r => setTimeout(r, 2000));
      }
    }

    return NextResponse.json(
      { error: "HF Space tidak membalas JSON setelah 3x percobaan" },
      { status: 502 }
    );

  } catch (error) {
    console.error("[PROXY] Gagal menghubungi HF Space:", error);
    return NextResponse.json({ error: "Gagal konek ke server AI" }, { status: 500 });
  }
}