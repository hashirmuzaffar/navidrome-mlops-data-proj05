"""
Navidrome - Data Generator
Simulates realistic user traffic hitting the feedback endpoint.
Generates synthetic production data calibrated from real 30Music distributions.
Run: python3 pipeline/data_generator.py --endpoint http://localhost:8000 --users 100 --events 1000
"""
import argparse, time, random, json, requests
import numpy as np
from datetime import datetime, timezone

# calibrated from 30Music paper + our dataset stats
ACTION_WEIGHTS = {
    "play":         0.60,
    "skip":         0.25,
    "like":         0.08,
    "repeat":       0.05,
    "playlist_add": 0.02,
}
SCORE_MAP = {
    "play": 1.0, "skip": -1.0, "like": 3.0,
    "repeat": 2.0, "playlist_add": 1.5
}

# realistic session lengths from 30Music (avg 11 events/session)
SESSION_LENGTH_MEAN = 11
SESSION_LENGTH_STD  = 4

def generate_user_id(i):
    return f"synthetic_user_{i:05d}"

def generate_song_id(song_pool):
    # long-tail weighted selection (20% songs = 80% plays)
    weights = np.random.pareto(1.5, len(song_pool))
    weights = weights / weights.sum()
    return np.random.choice(song_pool, p=weights)

def generate_session(user_id, song_pool):
    """Generate one listening session for a user."""
    session_len = max(1, int(np.random.normal(SESSION_LENGTH_MEAN, SESSION_LENGTH_STD)))
    events = []
    for _ in range(session_len):
        action = random.choices(
            list(ACTION_WEIGHTS.keys()),
            weights=list(ACTION_WEIGHTS.values())
        )[0]
        song_id = generate_song_id(song_pool)
        total_dur = random.uniform(120, 360)
        if action == "skip":
            listened = random.uniform(5, 30)
        elif action in ["like", "repeat"]:
            listened = total_dur * random.uniform(0.8, 1.0)
        else:
            listened = total_dur * random.uniform(0.4, 1.0)

        events.append({
            "user_id": user_id,
            "song_id": song_id,
            "action": action,
            "score": SCORE_MAP[action],
            "duration_listened": round(listened, 2),
            "total_duration": round(total_dur, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "synthetic"
        })
    return events

def hit_endpoint(endpoint, event, verbose=False):
    try:
        r = requests.post(
            f"{endpoint}/api/feedback",
            json=event,
            timeout=5
        )
        if verbose:
            print(f"  POST /api/feedback -> {r.status_code} | {event['user_id']} {event['action']} {event['song_id']}")
        return r.status_code == 200
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000",
                        help="FastAPI endpoint URL")
    parser.add_argument("--users", type=int, default=50,
                        help="Number of synthetic users")
    parser.add_argument("--events", type=int, default=500,
                        help="Total events to generate")
    parser.add_argument("--songs", type=int, default=1000,
                        help="Song pool size")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Delay between events in seconds")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"=== Navidrome Data Generator ===")
    print(f"Endpoint: {args.endpoint}")
    print(f"Users:    {args.users}")
    print(f"Events:   {args.events}")
    print(f"Songs:    {args.songs}")
    print(f"Delay:    {args.delay}s")
    print()

    # generate song pool (realistic IDs)
    song_pool = [f"FMA_{i:06d}" for i in range(args.songs)]

    total_sent = 0
    total_ok   = 0
    start_time = time.time()

    # production data intentionally differs from training:
    # - includes cold-start users (new users with no history)
    # - includes songs not in FMA training set
    # - session dropout (not all sessions complete)
    # - noise in action labels

    cold_start_users = int(args.users * 0.3)  # 30% cold start
    warm_users = args.users - cold_start_users

    print(f"User breakdown:")
    print(f"  warm users (has history): {warm_users}")
    print(f"  cold start users (new):   {cold_start_users}")
    print()

    while total_sent < args.events:
        # pick a random user
        user_idx = random.randint(0, args.users - 1)
        user_id = generate_user_id(user_idx)

        # cold start users only play 1-2 songs before dropping off
        is_cold = user_idx >= warm_users
        if is_cold:
            session_events = generate_session(user_id, song_pool)[:2]
        else:
            session_events = generate_session(user_id, song_pool)

        for event in session_events:
            if total_sent >= args.events:
                break
            ok = hit_endpoint(args.endpoint, event, args.verbose)
            total_sent += 1
            total_ok   += 1 if ok else 0

            if total_sent % 100 == 0:
                elapsed = time.time() - start_time
                rate = total_sent / elapsed
                print(f"  [{total_sent}/{args.events}] "
                      f"{rate:.1f} events/sec | "
                      f"success: {total_ok/total_sent*100:.1f}%")

            time.sleep(args.delay)

    elapsed = time.time() - start_time
    print(f"\n=== GENERATOR COMPLETE ===")
    print(f"  total sent:    {total_sent:,}")
    print(f"  success rate:  {total_ok/total_sent*100:.1f}%")
    print(f"  duration:      {elapsed:.1f}s")
    print(f"  rate:          {total_sent/elapsed:.1f} events/sec")

if __name__ == "__main__":
    main()
