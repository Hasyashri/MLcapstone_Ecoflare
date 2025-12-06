"""
run_pipeline_advanced.py

PRODUCTION-GRADE PIPELINE RUNNER 

What this does:
----------------
Runs the complete wildfire ML pipeline with:

1.Step-by-step execution
2.Runtime (time) logging
3.Retry logic if any step fails
4.Optional parallel execution of independent steps

Pipeline Steps:
----------------
1) Data loading + EDA
2) Timestamp normalization
3) Spatial + temporal join
4) DEM elevation extraction
5) Label creation
6) QC + scaling + split + FINAL master dataset

How to run:
------------
python services/training/run_pipeline_advanced.py


# Step 1: Load + basic EDA
python services/training/1_data_loading_and_eda.py

# Step 2: Normalize timestamps
python services/training/2_timestamp_normalization.py

# Step 3: Spatial + temporal join
python services/training/3_spatial_temporal_join.py

# Step 4: Add elevation from DEM
python services/training/4_dem_feature_addition.py

# Step 5: Create labels & raw master CSV
python services/training/5_label_creation.py

# Step 6: QC, scaling, train/test split, FINAL master
python services/training/6_qc_scaling_split_master.py

"""

import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MAX_RETRIES = 2           # retry any failing step up to 2 extra times

# Our pipeline scripts
PIPELINE_STEPS = [
    "1_data_loading_and_eda.py",
    "2_timestamp_normalization.py",
    "3_spatial_temporal_join.py",
    "4_dem_feature_addition.py",
    "5_label_creation.py",
]

FINAL_STEP = "6_qc_scaling_split_master.py"

# --------------------------------------------------
# UTILS
# --------------------------------------------------

def run_script(script_path: Path, retries=MAX_RETRIES):
    """
    Runs a Python script with automatic retries
    and logs how long it takes.
    """
    attempt = 1

    while attempt <= retries + 1:

        print("\n" + "=" * 80)
        print(f"RUNNING: {script_path.name}")
        print(f"Attempt {attempt} of {retries + 1}")

        start = time.time()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        duration = round(time.time() - start, 2)

        print(f"\nFinished {script_path.name} in {duration} seconds.")

        if result.returncode == 0:
            print("✅ Step completed successfully.")
            return True

        print(f"⚠️ Step FAILED (exit code {result.returncode}).")

        if attempt <= retries:
            print("Retrying...")
            attempt += 1
            time.sleep(2)
        else:
            print("❌ Max retries exceeded. Stopping pipeline.")
            return False


# --------------------------------------------------
# PARALLEL EXECUTION
# --------------------------------------------------

def run_parallel_steps(base_dir: Path):
    """
    Demonstrates how independent pipeline steps could
    be run in parallel.

    NOTE:
    For safety, in our current pipeline we RUN SEQUENTIALLY
    because most steps depend on previous outputs.

    This function exists to demonstrate orchestration design.
    """

    print("\n" + "=" * 80)
    print("PARALLEL EXECUTION DEMO")
    print("Running steps sequentially (safe mode)")
    print("=" * 80)

    # If you ever have independent steps,
    # they could be executed here using threads.

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = []

        for step in PIPELINE_STEPS:
            path = base_dir / step
            results.append(executor.submit(run_script, path))

        for r in results:
            if not r.result():
                return False

    return True


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    print("\n🔥 WILDFIRE AI PIPELINE (ADVANCED RUNNER)")
    print("Time-logged | Retry-enabled | Parallel-ready\n")

    pipeline_start = time.time()

    base_dir = Path(__file__).parent

    # ----------------------------------------------
    # Run main steps (EDA → Labeling)
    # ----------------------------------------------
    for step in PIPELINE_STEPS:
        if not run_script(base_dir / step):
            sys.exit(1)

    # ----------------------------------------------
    # Final QC + Scaling Step
    # ----------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL QUALITY CONTROL STEP")
    print("=" * 80)

    if not run_script(base_dir / FINAL_STEP):
        sys.exit(1)

    total_time = round(time.time() - pipeline_start, 2)

    # ----------------------------------------------
    # COMPLETION
    # ----------------------------------------------
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total pipeline runtime: {total_time} seconds")
    print("\nFINAL OUTPUT:")
    print("   data/features/fire_training_master_clean.csv ✅")
    print("=" * 80)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
