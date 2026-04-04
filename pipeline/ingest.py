"""
Navidrome Recommendation System - Ingest Pipeline
Streams FMA metadata + FMA small directly to Swift in chunks.
Run: python pipeline/ingest.py
"""

import os, json, hashlib, zipfile, io
import requests
import pandas as pd
from datetime import datetime, timezone
import chi

# ── config ─────────────────────────────────────────────────────
CONTAINER        = "navidrome-bucket-proj05"
SITE             = "CHI@UC"
PROJECT          = "CHI-251409"
CHUNK_SIZE       = 50 * 1024 * 1024
CHECKPOINT_FILE  = "/tmp/ingest_checkpoint.json"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SMALL_URL    = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
RUN_ID           = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

# ── swift ──────────────────────────────────────────────────────
chi.use_site(SITE)
chi.set("project_name", PROJECT)
conn = chi.connection()

def swift_upload(data: bytes, object_name: str):
    conn.object_store.upload_object(
        container=CONTAINER,
        name=object_name,
        data=data
    )
    print(f"  uploaded → {object_name} ({len(data)/1e6:.1f} MB)")

# ── checkpoint ─────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "fma_small_chunks": 0}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)

def already_done(cp, key):
    return key in cp["completed"]

def mark_done(cp, key):
    if key not in cp["completed"]:
        cp["completed"].append(key)
    save_checkpoint(cp)

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ══════════════════════════════════════════════════════════════
# STEP 1 — FMA metadata (358MB)
# ══════════════════════════════════════════════════════════════
def ingest_fma_metadata(cp):
    key = "fma_metadata"
    if already_done(cp, key):
        print("[STEP 1] already done, skipping")
        return

    print("\n[STEP 1] Downloading FMA metadata (358MB)...")
    r = requests.get(FMA_METADATA_URL, stream=True, timeout=120)
    r.raise_for_status()

    buf = io.BytesIO()
    downloaded = 0
    for chunk in r.iter_content(1024 * 1024):
        buf.write(chunk)
        downloaded += len(chunk)
        print(f"  {downloaded/1e6:.0f} MB", end="\r")
    print(f"\n  done: {downloaded/1e6:.0f} MB")

    buf.seek(0)
    swift_upload(buf.read(), "raw/fma/fma_metadata.zip")

    manifest = {
        "source": FMA_METADATA_URL,
        "sha256": sha256_bytes(buf.getvalue()),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "files": []
    }

    buf.seek(0)
    with zipfile.ZipFile(buf) as z:
        csv_files = [n for n in z.namelist() if n.endswith(".csv")]
        print(f"  found {len(csv_files)} CSV files")

        for name in csv_files:
            print(f"  processing {name}...")
            csv_bytes = z.read(name)
            swift_upload(csv_bytes, f"raw/fma/{name}")

            try:
                df = pd.read_csv(
                    io.BytesIO(csv_bytes),
                    index_col=0, header=[0, 1]
                )
                df.columns = ['_'.join(c).strip() for c in df.columns]
            except Exception:
                try:
                    df = pd.read_csv(io.BytesIO(csv_bytes))
                except Exception as e:
                    print(f"    could not parse {name}: {e}")
                    continue

            before = len(df)
            df = df.dropna(how="all")
            after = len(df)
            print(f"    {name}: {before} → {after} rows")

            pq_buf = io.BytesIO()
            df.to_parquet(pq_buf, index=True, engine="pyarrow")
            pq_buf.seek(0)
            swift_upload(
                pq_buf.read(),
                f"processed/fma/{name.replace('.csv', '.parquet')}"
            )

            manifest["files"].append({
                "name": name,
                "rows_raw": before,
                "rows_clean": after,
                "sha256": sha256_bytes(csv_bytes)
            })

    swift_upload(
        json.dumps(manifest, indent=2).encode(),
        "raw/fma/metadata_manifest.json"
    )
    mark_done(cp, key)
    print("[STEP 1] FMA metadata complete")

# ══════════════════════════════════════════════════════════════
# STEP 2 — FMA small (7.2GB) stream in chunks to Swift
# ══════════════════════════════════════════════════════════════
def ingest_fma_small(cp):
    key = "fma_small"
    if already_done(cp, key):
        print("[STEP 2] already done, skipping")
        return

    print("\n[STEP 2] Streaming FMA small (7.2GB) to Swift in 50MB chunks...")
    head = requests.head(FMA_SMALL_URL, timeout=30)
    total = int(head.headers.get("content-length", 0))
    print(f"  total: {total/1e9:.2f} GB")

    start_chunk = cp.get("fma_small_chunks", 0)
    if start_chunk > 0:
        print(f"  resuming from chunk {start_chunk}")

    r = requests.get(FMA_SMALL_URL, stream=True, timeout=600)
    r.raise_for_status()

    chunk_num = 0
    uploaded  = 0
    buf       = io.BytesIO()
    buf_size  = 0

    for piece in r.iter_content(chunk_size=1024 * 1024):
        buf.write(piece)
        buf_size += len(piece)
        uploaded += len(piece)

        if buf_size >= CHUNK_SIZE:
            if chunk_num >= start_chunk:
                buf.seek(0)
                swift_upload(
                    buf.read(),
                    f"raw/fma/fma_small/chunk_{chunk_num:05d}.bin"
                )
                cp["fma_small_chunks"] = chunk_num + 1
                save_checkpoint(cp)

            chunk_num += 1
            buf      = io.BytesIO()
            buf_size = 0
            pct = uploaded / total * 100 if total else 0
            print(f"  {uploaded/1e9:.2f}/{total/1e9:.2f} GB ({pct:.1f}%)")

    if buf_size > 0:
        buf.seek(0)
        swift_upload(
            buf.read(),
            f"raw/fma/fma_small/chunk_{chunk_num:05d}.bin"
        )
        chunk_num += 1

    swift_upload(
        json.dumps({
            "source": FMA_SMALL_URL,
            "total_bytes": uploaded,
            "total_chunks": chunk_num,
            "chunk_size_mb": 50,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID
        }, indent=2).encode(),
        "raw/fma/fma_small_manifest.json"
    )
    mark_done(cp, key)
    print(f"[STEP 2] FMA small complete — {chunk_num} chunks")

# ══════════════════════════════════════════════════════════════
# STEP 3 — compute audio features from echonest.csv
# ══════════════════════════════════════════════════════════════
def compute_features(cp):
    key = "features"
    if already_done(cp, key):
        print("[STEP 3] already done, skipping")
        return

    print("\n[STEP 3] Computing audio features...")

    for csv_name in ["raw/fma/fma_metadata/echonest.csv",
                     "raw/fma/fma_metadata/features.csv"]:
        try:
            raw = conn.object_store.get_object(
                container=CONTAINER, name=csv_name)
            df = pd.read_csv(
                io.BytesIO(raw), index_col=0, header=[0, 1])
            df.columns = ['_'.join(c).strip() for c in df.columns]
            print(f"  loaded {csv_name}: {df.shape}")
            break
        except Exception as e:
            print(f"  {csv_name} not found: {e}")
            df = None

    if df is None:
        print("  no feature file found, skipping")
        return

    keywords = ["tempo", "loudness", "key", "mode", "energy",
                "danceability", "chroma", "mfcc", "spectral"]
    core = [c for c in df.columns
            if any(k in c.lower() for k in keywords)][:20]
    print(f"  selected {len(core)} features")

    feat = df[core].copy()
    before = len(feat)
    feat = feat.dropna(thresh=max(1, len(core) // 2))
    print(f"  rows: {before} → {len(feat)} after null filter")

    for col in core:
        mn, mx = feat[col].min(), feat[col].max()
        feat[col] = (feat[col] - mn) / (mx - mn) if mx > mn else 0.0

    version = datetime.now(timezone.utc).strftime("%Y%m%d")
    pq_buf = io.BytesIO()
    feat.to_parquet(pq_buf, index=True, engine="pyarrow")
    pq_buf.seek(0)
    swift_upload(
        pq_buf.read(),
        f"features/song-audio/v{version}/embeddings.parquet"
    )
    swift_upload(
        json.dumps({
            "version": version,
            "n_songs": len(feat),
            "n_features": len(core),
            "features": core,
            "shape": list(feat.shape),
            "normalization": "min-max [0,1]",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID
        }, indent=2).encode(),
        f"features/song-audio/v{version}/feature_manifest.json"
    )
    mark_done(cp, key)
    print(f"[STEP 3] Features done — {feat.shape}")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # clear checkpoint for fresh run
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint cleared")

    print(f"=== Navidrome Ingest Pipeline | run {RUN_ID} ===")
    print(f"Container : {CONTAINER}")
    print(f"Site      : {SITE}")

    cp = load_checkpoint()

    ingest_fma_metadata(cp)
    ingest_fma_small(cp)
    compute_features(cp)

    # final manifest
    swift_upload(
        json.dumps({
            "run_id": RUN_ID,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "steps": cp["completed"],
            "container": CONTAINER,
            "status": "success"
        }, indent=2).encode(),
        f"raw/run_manifest_{RUN_ID}.json"
    )
    print("\n=== INGEST COMPLETE ===")
