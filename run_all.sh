#!/usr/bin/env bash
# Runs the whole project, one command per part. Needs: MySQL running + sql/00_create_database.sql executed once.
set -e
python -m src.part1_generate_data        # Part 1: fake data -> MySQL
python -m src.part2_pipeline             # Part 2: clean, features, snapshots
python -m src.part3_ml                   # Part 3: personas + risk model, adds them to snapshots
pytest -q                                # unit tests for the guardrails
echo "Now start the API (new terminal):  uvicorn src.vi.app:app --port 8000"
echo "Then evaluate:                     python -m src.vi.evaluate --save-samples"
