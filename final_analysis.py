import os
import wandb
import subprocess
from functools import lru_cache

# Set up output directories and paths
eval_dir = "eval_dir"
configs_dir = os.path.join(eval_dir, "configs")
results_dir = os.path.join(eval_dir, "results")
analysis_dir = os.path.join(eval_dir, "analysis")
os.makedirs(configs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)

# Get the run from wandb
@lru_cache
def get_run(run_id):
    api = wandb.Api()
    run = api.run(run_id)
    return run

# Get config from run
def get_run_config(run_id):
    run = get_run(run_id)
    return run.config

# Get name of run
def get_run_name(run_id):
    run = get_run(run_id)
    return run.name

# Function to download the checkpoint using W&B API
def download_checkpoint(run_id, artifact_name, checkpoint_path):
    run = get_run(run_id)
    artifact = run.use_artifact(artifact_name)
    artifact_dir = artifact.download()
    checkpoint_file = os.path.join(artifact_dir, "checkpoint.pth")  # Replace with the correct file name in the artifact

    # Move the downloaded checkpoint to the desired path
    os.rename(checkpoint_file, checkpoint_path)

# Function to write the config file
def write_config_file(config, config_path):
    with open(config_path, "w") as f:
        f.write(config)

# Function to run the test script and store the results
def run_test_script(config_path, checkpoint_path, result_path):
    subprocess.run(["python", "tools/test.py", config_path, checkpoint_path, "--out", result_path])

# Function to run further analysis and store the results
def run_analysis(config_path, result_path, analysis_output_dir):
    os.makedirs(analysis_output_dir, exist_ok=True)
    
    # Get FLOPs
    with open(os.path.join(analysis_output_dir, "get_flops.txt"), "w") as f:
        subprocess.run(["python", "tools/analysis_tools/get_flops.py", config_path], stdout=f)

    # COCO error analysis
    with open(os.path.join(analysis_output_dir, "coco_error_analysis.txt"), "w") as f:
        subprocess.run(["python", "tools/analysis_tools/coco_error_analysis.py", result_path, analysis_output_dir], stdout=f)

    # Benchmark
    with open(os.path.join(analysis_output_dir, "benchmark.txt"), "w") as f:
        subprocess.run(["python", "-m", "torch.distributed.launch", "--nproc_per_node=1", "--master_port=29500",
                        "tools/analysis_tools/benchmark.py", config_path], stdout=f)

# Iterate through the list of run ids
run_list = [
        ]

for run in run_list:
    run_id = run
    run_name = get_run_name(run_id)
    config = get_run_config(run_id)

    config_path = os.path.join(configs_dir, f"{run_name}.py")
    checkpoint_path = os.path.join(configs_dir, f"{run_name}.pth")
    result_path = os.path.join(results_dir, f"{run_name}.json")
    analysis_output_dir = os.path.join(analysis_dir, run_name)

    # Download checkpoint
    download_checkpoint(run_id, model.get('artifact_name', f'run_{run_id}_model:latest'), checkpoint_path)

    # Write config file
    write_config_file(config, config_path)

    # Run the test script and store the results
    run_test_script(config_path, checkpoint_path, result_path)

    # Run further analysis and store the results
    run_analysis(config_path, result_path, analysis_output_dir)

