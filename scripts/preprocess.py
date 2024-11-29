"""
Convenience wrapper around src/data/preprocessing.py.

Usage::

    python scripts/preprocess.py --json data/brats_ssa_2024_5fold.json \\
                                  --out_dir data/processed --workers 8
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.preprocessing import main

if __name__ == "__main__":
    main()
