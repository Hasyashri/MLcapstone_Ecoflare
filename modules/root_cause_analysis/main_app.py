# main_app.py (integration snippet)
import streamlit as st
from root_cause_ui import run_root_cause_ui
from fire_detection_module import get_live_detections  # from Module 1

if __name__ == "__main__":
    live_fires = get_live_detections()  # returns list of fire dicts
    run_root_cause_ui(live_fires)
