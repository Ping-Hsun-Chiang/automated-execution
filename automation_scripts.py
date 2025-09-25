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

OPERATION_NOTEBOOK = "simulation_univariate_compare_four_model_matern_gabor_tdf2"

# Change Parameter
PARAMETER_NAME = "orientation"
PARAMETER_VALUES = [0, 0.7854, 1.5708]
OTHER_PARAMETERS = {}

STOP_ON_ERROR = True
SHOW_PROGRESS = True


def print_progress(current, total, bar_length=40):
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percent = f"{100 * (current / float(total)):.0f}%"
    print(f'Progress: |{bar}| {current}/{total} ({percent})')


def generate_output_filename(base_name, param_name, param_value):
    return f"{base_name}_{param_name}_{param_value}.ipynb"


def execute_notebook_with_params(base_notebook_path, param_name, param_value, index):
    base_name = base_notebook_path.stem
    output_filename = generate_output_filename(base_name, param_name, param_value)
    output_path = PROJECT_DIR / output_filename

    parameters = OTHER_PARAMETERS.copy()
    parameters[param_name] = param_value

    if SHOW_PROGRESS:
        print(f"\n{'='*60}")
        print(f"[{index}/{len(parameters)}] Processing: {base_notebook_path.name}")
        print(f"{'='*60}")
        print_progress(index-1, len(parameters))
    
    start_time = time.time()

    with open(os.devnull, 'w') as devnull:
        original_stderr = sys.stderr
        sys.stderr = devnull

        pm.execute_notebook(
            input_path=str(base_notebook_path),
            output_path=str(output_path),
            parameters=parameters,
            kernel_name=KERNEL,
            cwd=str(PROJECT_DIR),
            log_output=False,
            progress_bar=False,
            request_save_on_cell_execute=True,
        )

        sys.stderr = original_stderr
    
    execution_time = time.time() - start_time
    print(f"✅ Success: {output_filename} ({execution_time:.1f}s)")
    return True, execution_time


# Operation Go and Summary
print("🚀 Starting Notebook Processing")
print(f"📅 Start Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📄 Operation : {OPERATION_NOTEBOOK}.ipynb")
print(f"🔧 Try Parameter : {PARAMETER_NAME}={PARAMETER_VALUES}")
print()

base_notebook_path = PROJECT_DIR / f"{OPERATION_NOTEBOOK}.ipynb"

total_time = 0
for i, param_value in enumerate(PARAMETER_VALUES, 1):
    success, exec_time = execute_notebook_with_params(base_notebook_path, PARAMETER_NAME, param_value, i)
    total_time += exec_time

print_progress(len(PARAMETER_VALUES), len(PARAMETER_VALUES))
print(f"⏱️ Total Time: {total_time:.1f}s")
print(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")