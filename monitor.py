#!/usr/bin/env python3
"""Table monitor — live state display with analysis."""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.watch import run_file, run_live

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Table monitor.")
    parser.add_argument("-f", "--file", help="Process a single screenshot.")
    parser.add_argument("--all", action="store_true", help="Show every capture.")
    args = parser.parse_args()

    if args.file:
        run_file(args.file)
    else:
        run_live(show_all=args.all)
