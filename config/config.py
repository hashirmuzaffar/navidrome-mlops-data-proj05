# Central config for all pipeline scripts

# Chameleon Swift
SWIFT_CONTAINER = "navidrome-bucket-proj05"
SITE = "CHI@TACC"
PROJECT_NAME = "CHI-251409"

# MSD
MSD_SUMMARY_URL = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_summary_file.h5"
MSD_TASTE_URL = "http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip"

# Local working dirs
RAW_DIR = "/tmp/proj05/raw"
FEATURES_DIR = "/tmp/proj05/features"
PROCESSED_DIR = "/tmp/proj05/processed"

# Audio features we extract from MSD
AUDIO_FEATURES = ["tempo", "loudness", "key", "mode", "time_signature", "energy", "danceability"]

# Interaction scoring
SCORE_MAP = {
    "play": 1.0,
    "like": 3.0,
    "skip": -1.0,
    "dislike": -2.0,
    "repeat": 2.0,
    "playlist_add": 1.5,
}

# Leakage prevention
EVAL_HOLDOUT_USER_FRACTION = 0.15  # 15% of users reserved for eval only
TRAIN_EVAL_GAP_DAYS = 1            # min days between train end and eval start
