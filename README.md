# TraceLens AI

> **Media DNA & Cross-Platform Intelligence Engine**

TraceLens AI is a production-ready, full-stack media intelligence platform designed for cybersecurity researchers, OSINT analysts, fact-checkers, and digital forensic investigators. It enables the extraction of unique "Media DNA Profiles" combining cryptographic hashes, three independent perceptual hashes (pHash, dHash, aHash), custom audio spectrogram fingerprints, metadata structures, and optional AI semantic CLIP embeddings. The similarity matching and lineage mapping engines track variants across compression, resizing, watermarking, cropping, and re-encoding.

---

## Technical Stack & Architecture

- **Frontend**: Next.js 15 (App Router, Tailwind CSS v4, TypeScript, Lucide React, Framer Motion)
- **Backend**: FastAPI (Python 3.12, Uvicorn)
- **Database**: SQLite (SQLAlchemy ORM, ready for PostgreSQL migration)
- **Media Processing**: OpenCV (keyframe extraction), PIL & imagehash (perceptual hashing), librosa & scipy (spectral chroma audio profiling)
- **AI Semantics**: HuggingFace Transformers (OpenAI CLIP CPU-optimized vision model, optional for 8GB RAM CPU configurations)
- **Report Generation**: ReportLab (dark-themed forensic PDF documents)

---

## Directory Structure

```
TraceLens AI/
├── backend/
│   ├── app/
│   │   ├── uploads/            # Ingested media assets
│   │   ├── keyframes/          # Extracted video keyframe thumbnails
│   │   ├── reports/            # Generated forensic PDF documents
│   │   ├── database.py         # SQLAlchemy SQLite session config
│   │   ├── models.py           # Case, MediaItem, Relationship models
│   │   ├── schemas.py          # Pydantic schemas
│   │   ├── dna_engine.py       # DNA calculations & integrity scoring
│   │   ├── phash_visualizer.py # 6-stage pHash steps base64 encoder
│   │   ├── video_analyzer.py   # OpenCV framing & ffmpeg audio extraction
│   │   ├── similarity_engine.py# Hamming, chroma correlation & origin ranking
│   │   ├── variant_generator.py# Ingestion variant generator (crop, watermark, etc.)
│   │   ├── report_generator.py # ReportLab PDF design builder
│   │   ├── seeder.py           # Programmatic database auto-seeder
│   │   └── main.py             # FastAPI entrypoint, endpoints, & CORS settings
│   ├── requirements.txt        # Python package manifests (CPU-optimized torch)
│   └── .env                    # Config variable flags
├── frontend/
│   ├── src/app/
│   │   ├── compare/            # Side-by-side DNA grid differentials
│   │   ├── media/[id]/         # Media Details, Keyframes, & SVG network graphs
│   │   ├── playground/         # Sliders for virtual digital adjustments
│   │   ├── upload/             # Real-time pipeline visualizer
│   │   ├── globals.css         # Cyber glow styles & animations
│   │   └── layout.tsx / page.tsx# Navigation headers & Dashboard metrics
│   ├── package.json            # React & Next.js libraries
│   └── tsconfig.json           # TypeScript build targets
└── README.md                   # Setup instructions and descriptions
```

---

## Core Features

1. **Dashboard Console**: Interactive widgets representing total indexed items, forensic cases, matching items, and average confidence levels alongside recent ingestion feeds.
2. **Investigation Cases**: Forensic partitioning grouping media items into active case directories (e.g. Case #2026-ALPHA).
3. **Real-time Pipeline Tracker**: Ingestion progress board visually sequencing file hashing, audio parsing, embedding extraction, and matching.
4. **Interactive 6-Stage pHash Visualizer**: Educational stepper displaying the original, grayscale, downsampled grid, 2D DCT heatmap, 8x8 low frequency extraction, and final binary pHash matrix.
5. **DNA Side-by-Side Comparator**: Comparison console highlighting matching/mismatching bits in a 64-bit grid (green/red) and outputting an explainable similarity report.
6. **Fingerprint Sandbox Playground**: Interactive slider sandbox allowing analysts to virtually apply crop, watermark, scale, and compression, and observe bit divergence in real-time.
7. **Lineage SVG Propagation Graphs**: Dynamic tree diagrams displaying variants radiating from the estimated primary source.
8. **Forensic PDF Reports**: Streamable dark-themed PDF documents containing Executive Summaries, Media DNA tables, forensics analysis, and risk scoring.

---

## Local Setup & Installation

### Prerequisite System Packages
- **Python**: Version `3.10` or higher (tested on `3.12.0`).
- **Node.js**: Version `18.0` or higher (tested on `24.14.1`).
- **FFmpeg**: Required on system PATH for video framerate keyframing and audio extraction. Ensure the `ffmpeg` command is executable in your terminal.

---

### Step 1: Start the Backend Server

1. Open a terminal and navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Build/initialize python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Boot the FastAPI web server using Uvicorn:
   ```bash
   python -m uvicorn app.main:app --reload --port 8000
   ```
   *Note: Upon startup, the database `tracelens.db` will be created automatically in the root of the backend folder, and **seeded with 15 mock media variants** (3 originals with 4 variations each) linked under Case #2026-ALPHA.*

---

### Step 2: Start the Frontend App

1. Open a separate terminal and navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Run the Next.js dev server:
   ```bash
   npm run dev
   ```
3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

---

## 8GB CPU-Only / Low-Spec Optimizations

To run the application with minimal RAM:
- Set `ENABLE_CLIP=false` in `backend/.env` to skip HuggingFace model downloads (~150MB). The backend will automatically fall back to zero-filled vectors and run matching via perceptual hashes and spectrogram peak mappings without crashing.
- Audio analysis loads files at a low sample rate (11kHz) and caps video audio clips at 2 minutes to keep RAM footprints below 500MB.

---

## Future-Ready Roadmap hooks
The architecture is prepared with internal modular routes to hook:
1. **Reddit & Telegram Ingestion**: Scrapers to ingest media from public channels directly into active investigation cases.
2. **Reverse Image Search**: Search indexed perceptual hashes using Hamming distance lookup indexes.
3. **Deepfake Visual Analysis**: Incorporate neural face-swapping detection modules.
4. **Threat Intelligence Feeds**: Integrate indicators of compromise (IoC) with media watchlists.
