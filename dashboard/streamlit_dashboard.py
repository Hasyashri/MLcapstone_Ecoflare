import os
import sys

# Fix module import paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st   # ← REQUIRED

from modules.fire_detection.fire_detection_logic import run_detection_voting
from modules.fire_spread_prediction.ml_integration import run_spread_prediction_pipeline
from modules.root_cause_analysis.root_cause_ui import render_root_cause_tab
