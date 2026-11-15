"""Uplift modeling and budget-constrained targeting for retention offers."""

import os
import sys

if sys.platform == "darwin":
    # LightGBM and scikit-learn each link an OpenMP runtime on macOS. If two copies end up
    # loaded, libomp aborts ("OMP: Error #15") although running both is harmless here.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
