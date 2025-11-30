import os
import sys

# Make project root importable (so 'modules' can be found)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

    
from modules.root_cause_analysis.root_cause_ui import render_root_cause_tab

# ...
tab1, tab2, tab3 = st.tabs(["Detection", "Spread Prediction", "Root Cause"])
# ...
with tab3:
    render_root_cause_tab()
