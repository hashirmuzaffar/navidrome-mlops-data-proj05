# Central config for all pipeline scripts

# Chameleon Swift
SWIFT_CONTAINER = "navidrome-bucket-proj05"
SITE            = "CHI@UC"
PROJECT_NAME    = "CHI-251409"

# FMA
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_MEDIUM_URL   = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"

# 30Music
THIRTYMUSIC_PATH = "/home/hm3680_nyu_edu/work/work/entities"

# Local working dirs (temp only — pipeline streams to Swift)
TMP_DIR = "/tmp/proj05"

# Audio features for BPR-kNN cold start
AUDIO_FEATURES = [
    "tempo", "loudness", "key", "mode",
    "energy", "danceability", "chroma_mean",
    "mfcc_mean", "spectral_centroid_mean"
]

# Interaction scoring for BPR-MF
SCORE_MAP = {
    "play":         1.0,
    "like":         3.0,
    "skip":        -1.0,
    "dislike":     -2.0,
    "repeat":       2.0,
    "playlist_add": 1.5,
    "favorite":     2.5,
}

# Leakage prevention
EVAL_HOLDOUT_USER_FRACTION = 0.15
TRAIN_EVAL_GAP_DAYS        = 1

# Validation thresholds
MIN_USER_INTERACTIONS = 5
MIN_SONG_PLAYS        = 1
CHUNK_SIZE_MB         = 50
