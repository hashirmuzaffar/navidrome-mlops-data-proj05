"""
Navidrome Recommendation System - Ingest Pipeline
Streams FMA metadata + FMA medium directly to Swift in chunks.
Run: python pipeline/ingest.py
"""

import os, sys, json, hashlib, zipfile, io
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
FMA_MEDIUM_URL   = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
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
    return {"completed": [], "fma_medium_chunks": 0}

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
# STEP 1 — FMA metadata zip (358MB)
# Extract CSVs, upload raw + parquet versions to Swift
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
        print(f"  {downloaded/1e6:.0f} MB downloaded", end="\r")
    print(f"\n  done: {downloaded/1e6:.0f} MB")

    # upload raw zip
    buf.seek(0)
    swift_upload(buf.read(), "raw/fma/fma_metadata.zip")

    # extract CSVs and upload individually
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
        print(f"  found {len(csv_files)} CSV files: {csv_files}")

        for name in csv_files:
            print(f"  processing {name}...")
            csv_bytes = z.read(name)

            # upload raw csv
            swift_upload(csv_bytes, f"raw/fma/{name}")

            # parse + validate + convert to parquet
            try:
                df = pd.read_csv(
                    io.BytesIO(csv_bytes),
                    index_col=0,
                    header=[0, 1]
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
            print(f"    {name}: {before} → {after} rows after null drop")

            pq_buf = io.BytesIO()
            df.to_parquet(pq_buf, index=True)
            pq_buf.seek(0)
            pq_name = f"processed/fma/{name.replace('.csv', '.parquet')}"
            swift_upload(pq_buf.read(), pq_name)

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
# STEP 2 — FMA medium zip (22GB)
# Stream in 50MB chunks directly to Swift — never fills disk
# Checkpoint aware — resumes if interrupted
# ══════════════════════════════════════════════════════════════
def ingest_fma_medium(cp):
    key = "fma_medium"
    if already_done(cp, key):
        print("[STEP 2] already done, skipping")
        return

    print("\n[STEP 2] Streaming FMA medium (22GB) to Swift in 50MB chunks...")
    head = requests.head(FMA_MEDIUM_URL, timeout=30)
    total = int(head.headers.get("content-length", 0))
    print(f"  total size: {total/1e9:.2f} GB")

    start_chunk = cp.get("fma_medium_chunks", 0)
    if start_chunk > 0:
        print(f"  resuming from chunk {start_chunk}")

    r = requests.get(FMA_MEDIUM_URL, stream=True, timeout=300)
    r.raise_for_status()

    chunk_num  = 0
    uploaded   = 0
    buf        = io.BytesIO()
    buf_size   = 0

    for piece in r.iter_content(chunk_size=1024 * 1024):
        buf.write(piece)
        buf_size += len(piece)
        uploaded += len(piece)

        if buf_size >= CHUNK_SIZE:
            if chunk_num >= start_chunk:
                buf.seek(0)
                object_name = f"raw/fma/fma_medium/chunk_{chunk_num:05d}.bin"
                swift_upload(buf.read(), object_name)
                cp["fma_medium_chunks"] = chunk_num + 1
                save_checkpoint(cp)

            chunk_num += 1
            buf = io.BytesIO()
            buf_size = 0
            pct = uploaded / total * 100 if total else 0
            print(f"  {uploaded/1e9:.2f} GB / {total/1e9:.2f} GB ({pct:.1f}%)")

    # flush last chunk
    if buf_size > 0:
        buf.seek(0)
        object_name = f"raw/fma/fma_medium/chunk_{chunk_num:05d}.bin"
        swift_upload(buf.read(), object_name)
        chunk_num += 1

    manifest = {
        "source": FMA_MEDIUM_URL,
        "total_bytes": uploaded,
        "total_chunks": chunk_num,
        "chunk_size_mb": CHUNK_SIZE // (1024 * 1024),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID
    }
    swift_upload(
        json.dumps(manifest, indent=2).encode(),
        "raw/fma/fma_medium_manifest.json"
    )
    mark_done(cp, key)
    print(f"[STEP 2] FMA medium complete — {chunk_num} chunks")

# ══════════════════════════════════════════════════════════════
# STEP 3 — compute audio features from echonest.csv
# Normalize, select core features, upload as parquet
# ══════════════════════════════════════════════════════════════
def compute_features(cp):
    key = "features"
    if already_done(cp, key):
        print("[STEP 3] already done, skipping")
        return

    print("\n[STEP 3] Computing audio features from echonest.csv...")

    # pull echonest.csv back from Swift
    try:
        raw = conn.object_store.get_object(
            container=CONTAINER,
            name="raw/fma/fma_metadata/echonest.csv"
        )
        df = pd.read_csv(io.BytesIO(raw), index_col=0, header=[0, 1])
        df.columns = ['_'.join(c).strip() for c in df.columns]
        print(f"  echonest.csv loaded: {df.shape}")
    except Exception as e:
        print(f"  echonest.csv not found ({e}), trying features.csv...")
        try:
            raw = conn.object_store.get_object(
                container=CONTAINER,
                name="raw/fma/fma_metadata/features.csv"
            )
            df = pd.read_csv(io.BytesIO(raw), index_col=0, header=[0, 1])
            df.columns = ['_'.join(c).strip() for c in df.columns]
            print(f"  features.csv loaded: {df.shape}")
        except Exception as e2:
            print(f"  no feature file found: {e2}")
            return

    # select relevant audio features for BPR-kNN
    keywords = ["tempo", "loudness", "key", "mode", "energy",
                 "danceability", "chroma", "mfcc", "spectral"]
    core = [c for c in df.columns
            if any(k in c.lower() for k in keywords)][:20]
    print(f"  selected {len(core)} features: {core[:5]}...")

    feat = df[core].copy()
    before = len(feat)
    feat = feat.dropna(thresh=len(core) // 2)
    print(f"  rows: {before} → {len(feat)} after null filter")

    # normalize each feature to [0,1]
    for col in core:
        mn, mx = feat[col].min(), feat[col].max()
        feat[col] = (feat[col] - mn) / (mx - mn) if mx > mn else 0.0

    version = datetime.now(timezone.utc).strftime("%Y%m%d")

    pq_buf = io.BytesIO()
    feat.to_parquet(pq_buf, index=True)
    pq_buf.seek(0)
    swift_upload(
        pq_buf.read(),
        f"features/song-audio/v{version}/embeddings.parquet"
    )

    feat_manifest = {
        "version": version,
        "n_songs": len(feat),
        "n_features": len(core),
        "features": core,
        "shape": list(feat.shape),
        "normalization": "min-max [0,1]",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID
    }
    swift_upload(
        json.dumps(feat_manifest, indent=2).encode(),
        f"features/song-audio/v{version}/feature_manifest.json"
    )
    mark_done(cp, key)
    print(f"[STEP 3] Features done — shape {feat.shape}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — run manifest
# ══════════════════════════════════════════════════════════════
def write_run_manifest(cp):
    print("\n[STEP 4] Writing run manifest...")
    manifest = {
        "run_id": RUN_ID,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "steps_completed": cp["completed"],
        "container": CONTAINER,
        "status": "success"
    }
    swift_upload(
        json.dumps(manifest, indent=2).encode(),
        f"raw/run_manifest_{RUN_ID}.json"
    )
    print(json.dumps(manifest, indent=2))

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"=== Navidrome Ingest Pipeline | run {RUN_ID} ===")
    print(f"Container : {CONTAINER}")
    print(f"Site      : {SITE}")

    cp = load_checkpoint()
    print(f"Checkpoint: {cp['completed']}")

    ingest_fma_metadata(cp)
    ingest_fma_medium(cp)
    compute_features(cp)
    write_run_manifest(cp)

    print("\n=== INGEST COMPLETE ===")
ENDOFFILE
