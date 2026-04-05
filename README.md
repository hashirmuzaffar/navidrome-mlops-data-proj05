# Navidrome Recommendation System — Data Pipeline

ECE-GY 9183 · ML Systems Design & Operations
Team: Yesha Vyas · Hashir Muzaffar · Vanshika Bagaria · Salauat Kakimzhanov

## Overview

Data pipeline for a BPR-MF + BPR-kNN music recommendation system
built on Navidrome, a self-hosted music streaming server.

## Datasets

| Dataset | Size | Content | Access |
|---------|------|---------|--------|
| FMA metadata | 358MB | 106K songs, audio features | Public download |
| FMA small | 7.2GB | 8K MP3 audio files | Public download |
| 30Music | ~1GB | 33K users, 56K playlists | Manual bootstrap |

## Bootstrap (one-time)

30Music requires manual upload due to academic registration requirement.
Upload ThirtyMusic.tar.gz to Swift before running the pipeline.
See docs/data_design_document.md for full instructions.

## Setup

    pip3 install requests pandas pyarrow python-chi fastapi uvicorn
    source ~/.chi_auth.sh

## Running the Pipeline

    make ingest          # Download FMA + 30Music, upload to Swift
    make validate        # Validate all datasets, reject bad rows
    make build-dataset   # Build versioned train/eval triplets
    make start-api       # Start feedback API on port 8000
    make generate        # Run data generator against API
    make all             # Run ingest + validate + build-dataset

## Object Storage Layout

    navidrome-bucket-proj05/
    raw/fma/                     FMA raw files + 147 chunks
    raw/30music/                 ThirtyMusic.tar.gz
    processed/fma/               9 parquet files
    processed/30music/           tracks chunks + users + playlists
    validated/                   validated versions of all datasets
    features/song-audio/         BPR-kNN audio embeddings
    datasets/v20260405-001/      versioned train/eval triplets + manifest
    production/events/           live feedback events from API

## Leakage Prevention

- Chronological split: train on past, eval on future
- 15% of users held out for eval only
- All sources tagged: fma, 30music, synthetic
- Full provenance in manifest.json per dataset version

## Pipeline Scripts

| Script | Purpose | Video |
|--------|---------|-------|
| pipeline/ingest.py | Download FMA to Swift | Video 1 |
| pipeline/parse_30music.py | Parse 30Music to Swift | Video 1 |
| pipeline/validate.py | Validate all datasets | Video 1 |
| pipeline/build_dataset.py | Build train/eval splits | Video 4 |
| pipeline/feedback_api.py | Online feature computation API | Video 3 |
| pipeline/data_generator.py | Synthetic traffic generator | Video 2 |
