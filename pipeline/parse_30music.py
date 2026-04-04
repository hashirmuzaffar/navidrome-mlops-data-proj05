"""
Parse 30Music idomaar files → clean parquet → upload to Swift
"""
import json, io, os
import pandas as pd
import chi

CONTAINER = "navidrome-bucket-proj05"
chi.use_site("CHI@UC")
chi.set("project_name", "CHI-251409")
conn = chi.connection()

def swift_upload(data: bytes, name: str):
    conn.object_store.upload_object(
        container=CONTAINER, name=name, data=data)
    print(f"  uploaded → {name} ({len(data)/1e6:.1f} MB)")

def parse_idomaar_line(line):
    parts = line.strip().split("\t")
    if len(parts) < 4:
        return None
    try:
        props = json.loads(parts[3])
        props["id"] = int(parts[1])
        props["timestamp"] = int(parts[2]) if parts[2] != "-1" else None
        if len(parts) > 4:
            props["relations"] = json.loads(parts[4])
        return props
    except Exception:
        return None

def parse_and_upload(path, entity_type, object_prefix):
    print(f"\nParsing {entity_type}...")
    records = []
    rejected = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = parse_idomaar_line(line)
            if row is None:
                rejected += 1
                continue
            records.append(row)

            if (i+1) % 100000 == 0:
                print(f"  parsed {i+1} lines...")

    print(f"  total: {len(records)} accepted, {rejected} rejected")

    df = pd.DataFrame(records)

    # drop all-null columns
    df = df.dropna(axis=1, how="all")
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")

    # upload as parquet
    pq_buf = io.BytesIO()
    df.to_parquet(pq_buf, index=False)
    pq_buf.seek(0)
    swift_upload(pq_buf.read(),
                 f"raw/30music/{object_prefix}.parquet")

    # upload manifest
    manifest = {
        "entity": entity_type,
        "rows": len(df),
        "rejected": rejected,
        "columns": list(df.columns),
        "source": path
    }
    swift_upload(
        json.dumps(manifest, indent=2).encode(),
        f"raw/30music/{object_prefix}_manifest.json"
    )
    return df

if __name__ == "__main__":
    base = "/home/hm3680_nyu_edu/work/work/entities"

    tracks_df = parse_and_upload(
        f"{base}/tracks.idomaar", "tracks", "tracks")

    users_df = parse_and_upload(
        f"{base}/users.idomaar", "users", "users")

    playlist_df = parse_and_upload(
        f"{base}/playlist.idomaar", "playlists", "playlists")

    print("\n=== 30Music parse complete ===")
    print(f"Tracks:    {len(tracks_df):,}")
    print(f"Users:     {len(users_df):,}")
    print(f"Playlists: {len(playlist_df):,}")
ENDOFFILE
