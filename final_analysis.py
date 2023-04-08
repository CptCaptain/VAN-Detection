import os
import json
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
def download_checkpoint(run_path, artifact_name, checkpoint_path):
    api = wandb.Api()
    run = get_run(run_path)
    artifact = api.artifact(f'{run_path.rsplit("/", 1)[0]}/{artifact_name}')
    artifact_dir = artifact.download()
    checkpoint_file = os.path.join(artifact_dir, "epoch_12.pth") # Replace with the correct file name in the artifact

    # Move the downloaded checkpoint to the desired path
    os.rename(checkpoint_file, checkpoint_path)


# Function to recursively convert JSON values to Python values
def json_to_python(obj):
    if isinstance(obj, dict):
        # special case for 'img_scale' key
        if 'img_scale' in obj:
            img_scale = obj['img_scale']
            obj['img_scale'] = [tuple(img_scale),]  # Convert to list[tuple[int, int]] format
        # special case for renamed models
        if obj.get('type') == 'VAN':
            obj['type'] = 'VAN_Official'
        return {k: json_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_to_python(elem) for elem in obj]
    else:
        return obj

# Function to write the config file
def write_config_file(config_dict, config_path):
    config_dict = json_to_python(config_dict)

    with open(config_path, "w") as f:
        for key, value in config_dict.items():
            if isinstance(value, str):
                f.write(f"{key} = '{value}'\n")
            else:
                f.write(f"{key} = {value}\n")

# Function to run the test script and store the results
def run_test_script(config_path, checkpoint_path, result_path):
    subprocess.run(["python", "test.py", config_path, checkpoint_path, "--out", result_path])

# Function to run further analysis and store the results
def run_analysis(config_path, result_path, analysis_output_dir):
    os.makedirs(analysis_output_dir, exist_ok=True)
    tool_path = 'mmdetection/tools/analysis_tools'
    
    # Get FLOPs
    print('Calculating complexity')
    with open(os.path.join(analysis_output_dir, "get_flops.txt"), "w") as f:
        subprocess.run(["python", f"{tool_path}/get_flops.py", config_path], stdout=f)

    # # COCO error analysis
    # this might need results in json format
    # with open(os.path.join(analysis_output_dir, "coco_error_analysis.txt"), "w") as f:
        # subprocess.run(["python", f"{tool_path}/coco_error_analysis.py", result_path, analysis_output_dir], stdout=f)

    # Benchmark
    print('Benchmarking')
    os.environ['LOCAL_RANK'] = "0"
    with open(os.path.join(analysis_output_dir, "benchmark.txt"), "w") as f:
        subprocess.run(["python", "-m", "torch.distributed.launch", "--nproc_per_node=1", "--master_port=29500",
                        f"{tool_path}/benchmark.py", config_path, checkpoint_path], stdout=f)

# Iterate through the list of run ids
run_list = [
        '111lxdne',
        '1vitd2f2',
        ]

for run_id in run_list:
    run_path = f'nkoch-aitastic/van-detection/{run_id}'
    run_name = get_run_name(run_path)
    config = get_run_config(run_path)

    config_path = os.path.join(configs_dir, f"{run_name}.py")
    checkpoint_path = os.path.join(configs_dir, f"{run_name}.pth")
    result_path = os.path.join(results_dir, f"{run_name}.pkl")
    analysis_output_dir = os.path.join(analysis_dir, run_name)

    # Download checkpoint
    download_checkpoint(run_path, f'run_{run_id}_model:latest', checkpoint_path)

    # Write config file
    write_config_file(config, config_path)

    # Run the test script and store the results
    if not os.path.exists(result_path):
        # only run test script if we don't have results already, it's expensive
        run_test_script(config_path, checkpoint_path, result_path)

    # Run further analysis and store the results
    run_analysis(config_path, result_path, analysis_output_dir)

