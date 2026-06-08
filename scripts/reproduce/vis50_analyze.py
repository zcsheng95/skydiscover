#!/usr/bin/env python3
"""Compatibility wrapper for the VIS bilevel experiment analyzer."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "experiments" / "vis50_bilevel_latest_only" / "analyze.py"
runpy.run_path(str(TARGET), run_name="__main__")
