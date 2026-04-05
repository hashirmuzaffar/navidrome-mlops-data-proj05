"""
Navidrome - Batch Dataset Builder
Compiles versioned train/eval datasets from production events + historical data.
Production data intentionally differs from training data (real serving challenges).
Run: source ~/.chi_auth.sh && python3 pipeline/build_dataset.py
"""
import os, json, subprocess, io, ast
import pandas as pd
import numpy as np
from datetime import datetime, timezone

CONTAINER    = "navidrome-bucket-proj05"
RUN_ID       = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
VERSION      = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}-001"
HOLDOUT_FRAC = 0.15
MAX_PER_USER = 50

AUTH_ARGS = [
    "--os-auth-url", os.environ["OS_AUTH_URL"],
    "--os-auth-type", "v3applicationcredential",
    "--os-application-credential-id", os.environ["OS_APPLICATION_CREDENTIAL_ID"],
    "--os-application-credential-secret", os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
]

def swift_run(args):
    return subprocess.run(["swift"] + AUTH_ARGS + args,
                         capture_output=True, text=True)

def swift_upload(local, name):
    swift_run(["upload", "--object-name", name, CONTAINER, local])
    print(f"  uploaded -> {name} ({os.path.getsize(local)/1e6:.1f} MB)")

def swift_upload_bytes(data, name):
    tmp = f"/tmp/ds_{RUN_ID}.bin"
    with open(tmp, "wb") as f:
        f.write(data)
    swift_upload(tmp, name)
    os.remove(tmp)

def swift_download(name, local):
    swift_run(["download", "--output", local, CONTAINER, name])

def list_objects(prefix):
    r = swift_run(["list", CONTAINER, "--prefix", prefix])
    return [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]

# ══════════════════════════════════════════════════════════════
# STEP 1 — Load 30Music playlist interactions (historical)
# ══════════════════════════════════════════════════════════════
def load_historical():
    print("\n[STEP 1] Loading 30Music playlist interactions...")
    local = "/tmp/playlists_val.parquet"
    swift_download("validated/30music/playlists.parquet", local)
    playlists = pd.read_parquet(local, engine="pyarrow")
    os.remove(local)

    rows = []
    for _, row in playlists.iterrows():
        try:
            relations = row["relations"]
            if isinstance(relations, str):
                relations = ast.literal_eval(relations)
            if not isinstance(relations, dict):
                continue
            subjects = relations.get("subjects", [])
            objects  = relations.get("objects", [])
            if not subjects or not objects:
                continue
            user_id = subjects[0].get("id")
            ts = pd.to_datetime(row.get("timestamp", 0), unit="s", errors="coerce")
            for obj in objects:
                song_id = obj.get("id")
                if user_id and song_id:
                    rows.append({
                        "user_id":   f"30m_{user_id}",
                        "song_id":   str(song_id),
                        "score":     1.5,
                        "action":    "playlist_add",
                        "timestamp": ts,
                        "source":    "30music_historical"
                    })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    print(f"  historical interactions: {len(df):,}")
    print(f"  unique users: {df['user_id'].nunique():,}")
    return df

# ══════════════════════════════════════════════════════════════
# STEP 2 — Load production events (from data generator)
# ══════════════════════════════════════════════════════════════
def load_production():
    print("\n[STEP 2] Loading production events from Swift...")
    objects = list_objects("production/events/")
    print(f"  found {len(objects)} production event files")

    if not objects:
        print("  no production events found — skipping")
        return pd.DataFrame()

    dfs = []
    for obj in objects:
        local = f"/tmp/prod_{len(dfs)}.parquet"
        swift_download(obj, local)
        try:
            df = pd.read_parquet(local, engine="pyarrow")
            dfs.append(df)
        except Exception as e:
            print(f"  skipping {obj}: {e}")
        finally:
            if os.path.exists(local):
                os.remove(local)

    if not dfs:
        return pd.DataFrame()

    prod = pd.concat(dfs, ignore_index=True)
    prod["timestamp"] = pd.to_datetime(prod["timestamp"], errors="coerce")
    prod["score"] = pd.to_numeric(prod["score"], errors="coerce")

    # production data intentionally differs from training:
    # includes cold-start users, skips, session dropouts
    print(f"  production interactions: {len(prod):,}")
    print(f"  unique users: {prod['user_id'].nunique():,}")
    print(f"  action distribution:")
    print(prod["action"].value_counts().to_string())
    return prod

# ══════════════════════════════════════════════════════════════
# STEP 3 — Combine + chronological split + generate triplets
# ══════════════════════════════════════════════════════════════
def build_triplets(historical_df, production_df):
    print("\n[STEP 3] Building dataset with chronological split...")
    np.random.seed(42)

    # combine sources
    if not production_df.empty:
        all_data = pd.concat([historical_df, production_df], ignore_index=True)
        print(f"  combined: {len(historical_df):,} historical + {len(production_df):,} production")
    else:
        all_data = historical_df.copy()
        print(f"  using historical only: {len(all_data):,} interactions")

    all_data["timestamp"] = pd.to_datetime(all_data["timestamp"], utc=True, errors="coerce")
    all_data = all_data.sort_values("timestamp").reset_index(drop=True)

    # user-level holdout — 15% reserved for eval only
    all_users = all_data["user_id"].unique()
    holdout = set(np.random.choice(
        all_users, size=int(len(all_users) * HOLDOUT_FRAC), replace=False))

    train_pool = all_data[~all_data["user_id"].isin(holdout)]
    eval_pool  = all_data[all_data["user_id"].isin(holdout)]

    # chronological split on train pool
    cutoff_idx = int(len(train_pool) * 0.8)
    if cutoff_idx < len(train_pool):
        train_cutoff = train_pool.iloc[cutoff_idx]["timestamp"]
        train_final  = train_pool[train_pool["timestamp"] <= train_cutoff]
        val_extra    = train_pool[train_pool["timestamp"] > train_cutoff]
        eval_combined = pd.concat([eval_pool, val_extra], ignore_index=True)
    else:
        train_final   = train_pool
        eval_combined = eval_pool

    print(f"  train: {len(train_final):,} | eval: {len(eval_combined):,}")
    print(f"  holdout users: {len(holdout):,}")

    all_songs = all_data["song_id"].unique()

    def make_triplets(data):
        triplets = []
        for user_id, group in data.groupby("user_id"):
            pos = list(group[group["score"] > 0]["song_id"].unique())
            pos_set = set(pos)
            neg_pool = [s for s in np.random.choice(
                all_songs, size=min(200, len(all_songs)), replace=False)
                if s not in pos_set]
            for p in pos[:MAX_PER_USER]:
                if neg_pool:
                    triplets.append({
                        "user_id":     user_id,
                        "pos_song_id": p,
                        "neg_song_id": np.random.choice(neg_pool)
                    })
        return pd.DataFrame(triplets)

    print("  generating train triplets...")
    train_t = make_triplets(train_final)
    print(f"  train triplets: {len(train_t):,}")

    print("  generating eval triplets...")
    eval_t = make_triplets(eval_combined)
    print(f"  eval triplets: {len(eval_t):,}")

    return train_t, eval_t, train_final, eval_combined

# ══════════════════════════════════════════════════════════════
# STEP 4 — Upload versioned dataset
# ══════════════════════════════════════════════════════════════
def upload_dataset(train_t, eval_t, train_df, eval_df, historical_df, production_df):
    print(f"\n[STEP 4] Uploading dataset {VERSION}...")
    prefix = f"datasets/{VERSION}"

    tmp = f"/tmp/train_{VERSION}.parquet"
    train_t.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/train_triplets.parquet")
    os.remove(tmp)

    tmp = f"/tmp/eval_{VERSION}.parquet"
    eval_t.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/eval_triplets.parquet")
    os.remove(tmp)

    manifest = {
        "version_id":            VERSION,
        "run_id":                RUN_ID,
        "created_at":            datetime.now(timezone.utc).isoformat(),
        "leakage_check":         "chronological_strict_user_holdout",
        "holdout_user_fraction": HOLDOUT_FRAC,
        "max_triplets_per_user": MAX_PER_USER,
        "neg_sampling":          "random_from_non_interacted",
        "sources": {
            "historical": {
                "name":         "30music_playlists",
                "interactions": len(historical_df),
                "users":        int(historical_df["user_id"].nunique()),
                "note":         "Real Last.fm playlist data 2011-2014"
            },
            "production": {
                "name":         "feedback_api_synthetic",
                "interactions": len(production_df) if not production_df.empty else 0,
                "users":        int(production_df["user_id"].nunique()) if not production_df.empty else 0,
                "note":         "Synthetic traffic with cold-start users and session noise"
            }
        },
        "train_interactions": len(train_df),
        "eval_interactions":  len(eval_df),
        "train_triplets":     len(train_t),
        "eval_triplets":      len(eval_t),
        "train_users":        int(train_df["user_id"].nunique()),
        "eval_users":         int(eval_df["user_id"].nunique()),
        "candidate_selection": {
            "positive": "playlist_add score > 0",
            "negative": "random sample from non-interacted songs",
            "rationale": "BPR requires implicit negative sampling"
        }
    }

    swift_upload_bytes(
        json.dumps(manifest, indent=2).encode(),
        f"{prefix}/manifest.json"
    )
    return manifest

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"=== Navidrome Build Dataset | run {RUN_ID} ===")
    print(f"Version: {VERSION}")

    historical_df  = load_historical()
    production_df  = load_production()
    train_t, eval_t, train_df, eval_df = build_triplets(historical_df, production_df)
    manifest = upload_dataset(train_t, eval_t, train_df, eval_df,
                              historical_df, production_df)

    print("\n=== BUILD DATASET COMPLETE ===")
    print(json.dumps(manifest, indent=2))
