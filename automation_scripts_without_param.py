# Library
import os
import sys
from pathlib import Path
import papermill as pm
from datetime import datetime
import time
import warnings

# Global Environment Setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Operation Setting
KERNEL = "air-pollution"
PROJECT_DIR = Path(__file__).resolve().parent


OPERATION_NOTEBOOK = [
    "simulation_univariate_non_eggholder_gaussian0.1",
    "simulation_univariate_non_eggholder_gaussian0.05",
    "simulation_univariate_non_eggholder_tdf4"
]

STOP_ON_ERROR = True
SHOW_PROGRESS = True


def print_progress(current, total, bar_length=40):
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percent = f"{100 * (current / float(total)):.0f}%"
    print(f'Progress: |{bar}| {current}/{total} ({percent})')


def execute_notebook(notebook_name, index, total):
    notebook_path = PROJECT_DIR / f"{notebook_name}.ipynb"
    
    if SHOW_PROGRESS:
        print(f"\n{'='*60}")
        print(f"[{index}/{total}] Processing: {notebook_name}.ipynb")
        print(f"{'='*60}")
        print_progress(index-1, total)
    
    start_time = time.time()
   
    with open(os.devnull, 'w') as devnull:
        original_stderr = sys.stderr
        sys.stderr = devnull
        
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(notebook_path),
            kernel_name=KERNEL,
            cwd=str(PROJECT_DIR),
            log_output=False,
            progress_bar=False,
            request_save_on_cell_execute=True,
        )
            
        sys.stderr = original_stderr
        
    execution_time = time.time() - start_time
    print(f"✅ Success: {notebook_name}.ipynb ({execution_time:.1f}s)")
    return execution_time


# Operation Go and Summary
print("🚀 Starting Notebook Processing")
print(f"📅 Start Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Total Notebooks: {len(OPERATION_NOTEBOOK)}")
print()

total_time = 0

for i, notebook_name in enumerate(OPERATION_NOTEBOOK, 1):
    exec_time = execute_notebook(notebook_name, i, len(OPERATION_NOTEBOOK))
    total_time += exec_time

print_progress(len(OPERATION_NOTEBOOK), len(OPERATION_NOTEBOOK))
print(f"\n{'='*60}")
print(f"⏱️ Total Time: {total_time:.1f}s")
print(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")