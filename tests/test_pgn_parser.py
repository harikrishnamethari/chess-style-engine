"""
tests/test_pgn_parser.py
Unit tests for the PGN parser utility.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.ml.pgn_parser import extract_game_metadata, extract_history_features

SAMPLE_PGN = """[Event "Test"]
[White "AlphaA"]
[Black "BetaB"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 1-0
"""

def test_metadata_extraction():
    meta = extract_game_metadata(SAMPLE_PGN)
    assert meta is not None
    assert meta["white"] == "AlphaA"
    assert meta["black"] == "BetaB"
    assert meta["result"] == "1-0"

def test_history_features_shape():
    features = extract_history_features(SAMPLE_PGN, max_games=10)
    assert len(features) == 10, "Should pad to 10 entries"
    for f in features:
        assert len(f) == 3, "Each feature vector must be length 3"

def test_history_features_range():
    features = extract_history_features(SAMPLE_PGN, max_games=10)
    for cp, cap, cas in features:
        assert 0 < cp <= 100, f"CP loss proxy out of range: {cp}"
        assert 0 <= cap <= 1, f"Capture rate out of range: {cap}"
        assert 0 < cas <= 60, f"Castling timing out of range: {cas}"

def test_invalid_pgn():
    meta = extract_game_metadata("this is not a pgn")
    assert meta is None
