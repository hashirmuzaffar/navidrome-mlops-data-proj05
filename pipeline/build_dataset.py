"""
Navidrome - Batch Dataset Builder
Compiles versioned training and evaluation datasets.
Enforces chronological split — no data leakage.
Run: source ~/.chi_auth.sh && python3 pipeline/build_dataset.py
"""
import os, json, subprocess, io
import pandas as pd
import numpy as np
from datetime import datetime, timezone

CONTAINER    = "navidrome-bucket-proj05"
RUN_ID       = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
VERSION      = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}-001"
HOLDOUT_FRAC = 0.15
SCORE_MAP    = {"play": 1.0, "like": 3.0, "skip": -1.0,
                "dislike": -2.0, "repeat": 2.0, "playlist_add": 1.5}

AUTH_ARGS = [
    "--os-auth-url", os.environ["OS_AUTH_URL"],
    "--os-auth-type", "v3applicationcredential",
    "--os-application-credential-id", os.environ["OS_APPLICATION_CREDENTIAL_ID"],
    "--os-application-credential-secret", os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
]

def swift_download(name, local):
    r = subprocess.run(["swift"] + AUTH_ARGS + [
        "download", "--output", local, CONTAINER, name
    ], capture_output=True, text=True)
    return r.returncode == 0

def swift_upload(local, name):
    subprocess.run(["swift"] + AUTH_ARGS + [
        "upload", "--object-name", name, CONTAINER, local
    ], capture_output=True)
    size = os.path.getsize(local)
    print(f"  uploaded -> {name} ({size/1e6:.1f} MB)")

def swift_upload_bytes(data, name):
    tmp = f"/tmp/ds_tmp_{RUN_ID}.bin"
    with open(tmp, "wb") as f:
        f.write(data)
    swift_upload(tmp, name)
    os.remove(tmp)

def list_objects(prefix):
    r = subprocess.run(["swift"] + AUTH_ARGS + [
        "list", CONTAINER, "--prefix", prefix
    ], capture_output=True, text=True)
    return [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]

# ══════════════════════════════════════════════════════════════
# STEP 1 — Load 30Music interactions from validated chunks
# ══════════════════════════════════════════════════════════════
def load_interactions():
    print("\n[STEP 1] Loading 30Music interactions...")
    chunks = list_objects("validated/30music/tracks/")
    print(f"  found {len(chunks)} validated track chunks")

    dfs = []
    for i, chunk in enumerate(chunks):
        local = f"/tmp/chunk_{i}.parquet"
        swift_download(chunk, local)
        df = pd.read_parquet(local, engine="pyarrow")
        df["playcount"] = pd.to_numeric(df["playcount"], errors="coerce")
        dfs.append(df[["id", "name", "playcount", "timestamp"]].copy())
        os.remove(local)
        if (i+1) % 10 == 0:
            print(f"  loaded {i+1}/{len(chunks)} chunks...")

    interactions = pd.concat(dfs, ignore_index=True)
    interactions = interactions.dropna(subset=["id", "playcount"])
    interactions = interactions[interactions["playcount"] > 0]
    print(f"  total interactions: {len(interactions):,}")
    return interactions

# ══════════════════════════════════════════════════════════════
# STEP 2 — Load playlists as user-track interactions
# ══════════════════════════════════════════════════════════════
def load_playlist_interactions():
    print("\n[STEP 2] Loading playlist interactions...")
    local = "/tmp/playlists_val.parquet"
    swift_download("validated/30music/playlists.parquet", local)
    playlists = pd.read_parquet(local, engine="pyarrow")
    os.remove(local)

    rows = []
    for _, row in playlists.iterrows():
        try:
            relations = row["relations"]
            if isinstance(relations, str):
                import ast
                relations = ast.literal_eval(relations)
            if not isinstance(relations, dict):
                continue
            subjects = relations.get("subjects", [])
            objects = relations.get("objects", [])
            if not subjects or not objects:
                continue
            user_id = subjects[0].get("id")
            for obj in objects:
                song_id = obj.get("id")
                if user_id and song_id:
                    rows.append({
                        "user_id": str(user_id),
                        "song_id": str(song_id),
                        "score": SCORE_MAP["playlist_add"],
                        "action": "playlist_add",
                        "timestamp": pd.to_datetime(row.get("timestamp", 0), unit="s", errors="coerce"),
                        "source": "30music_playlist"
                    })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    print(f"  playlist interactions: {len(df):,}")
    print(f"  unique users: {df['user_id'].nunique():,}")
    print(f"  unique songs: {df['song_id'].nunique():,}")
    return df

# ══════════════════════════════════════════════════════════════
# STEP 3 — Generate BPR triplets with chronological split
# ══════════════════════════════════════════════════════════════
def build_triplets(interactions_df):
    print("\n[STEP 3] Building BPR triplets...")

    df = interactions_df.copy()

    # sort by timestamp — CRITICAL for leakage prevention
    df = df.sort_values("timestamp").reset_index(drop=True)

    # holdout 15% of users for eval only — never appear in train
    all_users = df["user_id"].unique()
    np.random.seed(42)
    holdout_users = set(np.random.choice(
        all_users,
        size=int(len(all_users) * HOLDOUT_FRAC),
        replace=False
    ))
    print(f"  total users: {len(all_users):,}")
    print(f"  holdout users (eval only): {len(holdout_users):,}")

    train_df = df[~df["user_id"].isin(holdout_users)].copy()
    eval_df  = df[df["user_id"].isin(holdout_users)].copy()

    # chronological split on train users too
    # use first 80% of timestamps for train, last 20% for validation
    if "timestamp" in train_df.columns and train_df["timestamp"].notna().any():
        cutoff_idx = int(len(train_df) * 0.8)
        train_cutoff = train_df.iloc[cutoff_idx]["timestamp"]
        print(f"  train cutoff: {train_cutoff}")
        train_final = train_df[train_df["timestamp"] <= train_cutoff]
        val_from_train = train_df[train_df["timestamp"] > train_cutoff]
        eval_combined = pd.concat([eval_df, val_from_train], ignore_index=True)
    else:
        train_final = train_df
        eval_combined = eval_df

    print(f"  train interactions: {len(train_final):,}")
    print(f"  eval interactions:  {len(eval_combined):,}")

    # generate triplets (user, pos_song, neg_song)
    def make_triplets(data, n_neg=1):
        all_songs = data["song_id"].unique()
        triplets = []
        for user_id, group in data.groupby("user_id"):
            pos_songs = set(group[group["score"] > 0]["song_id"].tolist())
            neg_songs = set(group[group["score"] < 0]["song_id"].tolist())
            unseen = set(all_songs) - pos_songs - neg_songs

            for pos in pos_songs:
                # sample negatives from explicit negatives or unseen songs
                neg_pool = list(neg_songs) if neg_songs else list(unseen)
                if not neg_pool:
                    continue
                for _ in range(n_neg):
                    neg = np.random.choice(neg_pool)
                    triplets.append({
                        "user_id": user_id,
                        "pos_song_id": pos,
                        "neg_song_id": neg
                    })
        return pd.DataFrame(triplets)

    print("  generating train triplets...")
    train_triplets = make_triplets(train_final)
    print(f"  train triplets: {len(train_triplets):,}")

    print("  generating eval triplets...")
    eval_triplets = make_triplets(eval_combined)
    print(f"  eval triplets: {len(eval_triplets):,}")

    return train_triplets, eval_triplets, train_final, eval_combined

# ══════════════════════════════════════════════════════════════
# STEP 4 — Upload versioned dataset to Swift
# ══════════════════════════════════════════════════════════════
def upload_dataset(train_triplets, eval_triplets, train_df, eval_df):
    print(f"\n[STEP 4] Uploading versioned dataset {VERSION}...")

    prefix = f"datasets/{VERSION}"

    # train triplets
    tmp = f"/tmp/train_triplets_{VERSION}.parquet"
    train_triplets.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/train_triplets.parquet")
    os.remove(tmp)

    # eval triplets
    tmp = f"/tmp/eval_triplets_{VERSION}.parquet"
    eval_triplets.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/eval_triplets.parquet")
    os.remove(tmp)

    # manifest
    manifest = {
        "version_id": VERSION,
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "leakage_check": "chronological_strict",
        "holdout_user_fraction": HOLDOUT_FRAC,
        "train_interactions": len(train_df),
        "eval_interactions": len(eval_df),
        "train_triplets": len(train_triplets),
        "eval_triplets": len(eval_triplets),
        "train_users": train_df["user_id"].nunique(),
        "eval_users": eval_df["user_id"].nunique(),
        "sources": ["30music_playlist"],
        "notes": "Chronological split. 15% of users held out for eval only."
    }
    swift_upload_bytes(
        json.dumps(manifest, indent=2).encode(),
        f"{prefix}/manifest.json"
    )
    print(f"\n  dataset version: {VERSION}")
    print(f"  train triplets: {len(train_triplets):,}")
    print(f"  eval triplets:  {len(eval_triplets):,}")
    return manifest

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"=== Navidrome Build Dataset | run {RUN_ID} ===")
    print(f"Version: {VERSION}")
    print(f"Holdout fraction: {HOLDOUT_FRAC}")

    playlist_interactions = load_playlist_interactions()
    train_triplets, eval_triplets, train_df, eval_df = build_triplets(playlist_interactions)
    manifest = upload_dataset(train_triplets, eval_triplets, train_df, eval_df)

    print("\n=== BUILD DATASET COMPLETE ===")
    print(json.dumps(manifest, indent=2))
