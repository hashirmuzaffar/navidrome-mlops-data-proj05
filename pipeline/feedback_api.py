"""
Navidrome - Online Feature Computation + Feedback API
Receives user events, logs to Swift, updates online features.
Run: source ~/.chi_auth.sh && uvicorn pipeline.feedback_api:app --host 0.0.0.0 --port 8000
"""
import os, json, io, subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Navidrome Feedback API")

CONTAINER = "navidrome-bucket-proj05"
SCORE_MAP = {
    "play": 1.0, "skip": -1.0, "like": 3.0,
    "repeat": 2.0, "playlist_add": 1.5, "dislike": -2.0
}

AUTH_ARGS = [
    "--os-auth-url", os.environ.get("OS_AUTH_URL", ""),
    "--os-auth-type", "v3applicationcredential",
    "--os-application-credential-id", os.environ.get("OS_APPLICATION_CREDENTIAL_ID", ""),
    "--os-application-credential-secret", os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET", ""),
]

# in-memory event buffer — flushed to Swift every 100 events
event_buffer = []
FLUSH_EVERY  = 100
flush_count  = 0

class FeedbackEvent(BaseModel):
    user_id: str
    song_id: str
    action: str
    score: Optional[float] = None
    duration_listened: Optional[float] = None
    total_duration: Optional[float] = None
    timestamp: Optional[str] = None
    source: Optional[str] = "live"

class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 20

def swift_upload_bytes(data: bytes, object_name: str):
    tmp = f"/tmp/api_tmp_{datetime.now().strftime('%H%M%S%f')}.bin"
    with open(tmp, "wb") as f:
        f.write(data)
    subprocess.run(
        ["swift"] + AUTH_ARGS + [
            "upload", "--object-name", object_name,
            CONTAINER, tmp
        ],
        capture_output=True
    )
    os.remove(tmp)

def flush_buffer():
    global event_buffer, flush_count
    if not event_buffer:
        return
    df = pd.DataFrame(event_buffer)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    object_name = f"production/events/events_{ts}_batch{flush_count:04d}.parquet"
    swift_upload_bytes(buf.getvalue(), object_name)
    print(f"Flushed {len(event_buffer)} events -> {object_name}")
    event_buffer = []
    flush_count += 1

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "buffer_size": len(event_buffer)
    }

@app.post("/api/feedback")
def receive_feedback(event: FeedbackEvent):
    global event_buffer

    # compute score if not provided
    if event.score is None:
        event.score = SCORE_MAP.get(event.action, 1.0)

    # compute duration ratio
    duration_ratio = None
    if event.duration_listened and event.total_duration and event.total_duration > 0:
        duration_ratio = event.duration_listened / event.total_duration

    row = {
        "user_id": event.user_id,
        "song_id": event.song_id,
        "action": event.action,
        "score": event.score,
        "duration_listened": event.duration_listened,
        "total_duration": event.total_duration,
        "duration_ratio": duration_ratio,
        "timestamp": event.timestamp or datetime.now(timezone.utc).isoformat(),
        "source": event.source,
        "ingested_at": datetime.now(timezone.utc).isoformat()
    }

    event_buffer.append(row)

    # flush to Swift every FLUSH_EVERY events
    if len(event_buffer) >= FLUSH_EVERY:
        flush_buffer()

    return {
        "status": "accepted",
        "user_id": event.user_id,
        "song_id": event.song_id,
        "score": event.score,
        "buffer_size": len(event_buffer)
    }

@app.get("/api/recommendations")
def get_recommendations(user_id: str, top_k: int = 20):
    """
    Online feature computation path for real-time inference.
    Loads audio embeddings from Swift, computes cosine similarity.
    Cold-start: uses BPR-kNN (audio features only).
    """
    # load audio embeddings from Swift
    tmp = "/tmp/embeddings_cache.csv"
    if not os.path.exists(tmp):
        subprocess.run(
            ["swift"] + AUTH_ARGS + [
                "download", "--output", tmp,
                CONTAINER,
                "features/song-audio/v20260405/embeddings_audio_features.csv"
            ],
            capture_output=True
        )

    if not os.path.exists(tmp):
        return {"error": "embeddings not found", "user_id": user_id}

    embeddings = pd.read_csv(tmp, index_col=0)
    embeddings = embeddings.apply(pd.to_numeric, errors="coerce").dropna()

    # get user history from buffer
    user_events = [e for e in event_buffer if e["user_id"] == user_id]
    n_interactions = len(user_events)

    if n_interactions < 5:
        # cold start — BPR-kNN: recommend most popular songs
        # in production this would use audio similarity
        mode = "cold_start_bpr_knn"
        liked_songs = [e["song_id"] for e in user_events if e["score"] > 0]

        if liked_songs and liked_songs[0] in embeddings.index:
            # find similar songs using cosine similarity
            seed_vec = embeddings.loc[liked_songs[0]].values
            sims = embeddings.values @ seed_vec
            norms = np.linalg.norm(embeddings.values, axis=1) * np.linalg.norm(seed_vec)
            norms[norms == 0] = 1
            cos_sims = sims / norms
            top_idx = np.argsort(cos_sims)[::-1][:top_k]
            recommendations = [
                {"song_id": str(embeddings.index[i]),
                 "score": float(cos_sims[i]),
                 "rank": rank+1}
                for rank, i in enumerate(top_idx)
            ]
        else:
            # no seed song — return top songs by index
            recommendations = [
                {"song_id": str(sid), "score": 1.0, "rank": i+1}
                for i, sid in enumerate(embeddings.index[:top_k])
            ]
    else:
        # warm user — simplified BPR-MF (dot product on user embedding)
        mode = "warm_bpr_mf"
        liked = [e["song_id"] for e in user_events if e["score"] > 0]
        disliked = [e["song_id"] for e in user_events if e["score"] < 0]

        # build user embedding as mean of liked song embeddings
        liked_vecs = [embeddings.loc[s].values
                      for s in liked if s in embeddings.index]
        if liked_vecs:
            user_vec = np.mean(liked_vecs, axis=0)
            scores = embeddings.values @ user_vec
            top_idx = np.argsort(scores)[::-1][:top_k + len(disliked)]
            recommendations = []
            for i in top_idx:
                sid = str(embeddings.index[i])
                if sid not in disliked and len(recommendations) < top_k:
                    recommendations.append({
                        "song_id": sid,
                        "score": float(scores[i]),
                        "rank": len(recommendations)+1
                    })
        else:
            recommendations = []
        mode = "warm_bpr_mf"

    return {
        "user_id": user_id,
        "n_interactions": n_interactions,
        "mode": mode,
        "top_k": top_k,
        "recommendations": recommendations,
        "computed_at": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/stats")
def stats():
    return {
        "buffer_size": len(event_buffer),
        "flush_count": flush_count,
        "total_events_processed": flush_count * FLUSH_EVERY + len(event_buffer)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
