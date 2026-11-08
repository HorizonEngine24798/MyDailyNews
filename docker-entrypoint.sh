#!/bin/sh
set -e

# If neither config.recommended.json nor config.local.json exists, seed config.local.json from config.example.json
if [ ! -f "config.recommended.json" ] && [ ! -f "config.local.json" ]; then
    echo "[MyDailyNews Docker] No config file found. Initializing config.local.json from config.example.json..."
    cp config.example.json config.local.json
fi

# Ensure output, state, and cache directories exist
mkdir -p output state/memory .cache/mydailynews

exec "$@"
