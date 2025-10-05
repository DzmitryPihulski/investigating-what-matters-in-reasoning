import os
from pathlib import Path

import yaml

os.makedirs("workfolders", exist_ok=True)
os.makedirs("job_logs", exist_ok=True)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

slurm_template = f"""#!/bin/bash
#SBATCH --job-name={config["slurm_config"]["job-name"]}
#SBATCH --time={config["slurm_config"]["time"]}
#SBATCH --nodes={config["slurm_config"]["nodes"]}
#SBATCH --exclude={config["slurm_config"]["exclude"]}
#SBATCH --gres={config["slurm_config"]["gres"]}
#SBATCH --ntasks-per-node={config["slurm_config"]["ntasks-per-node"]}
#SBATCH --cpus-per-task={config["slurm_config"]["cpus-per-task"]}
#SBATCH --mem={config["slurm_config"]["mem"]}
#SBATCH -p {config["slurm_config"]["partition"]}
#SBATCH --verbose
#SBATCH --output={config["slurm_config"]["output"]}
#SBATCH --error={config["slurm_config"]["error"]}

echo "SLURM_JOB_ID $SLURM_JOB_ID"
echo "START TIME: $(date)"

export NUM_CPUS_PER_NODE={config["slurm_config"]["cpus-per-task"]}
export NUM_GPUS_PER_NODE={config["slurm_config"]["gres"].split(":")[config["slurm_config"]["gres"].split(":").index("hopper") + 1].split(",")[0]}
export NUM_NODES={config["slurm_config"]["nodes"]}
export TENSOR_PARALLEL_SIZE=16
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
cd $TMPDIR_LOCAL
mkdir -p .cache/pip
mkdir -p .cache/huggingface/datasets
mkdir -p .cache/triton/.cache
mkdir -p .cache/triton/.dump
mkdir -p .cache/vllm
mkdir -p wandb/cache
mkdir -p wandb/config

export PIP_CACHE_DIR=$TMPDIR_LOCAL/.cache/pip

export TRITON_HOME=$TMPDIR_LOCAL/.cache/triton
export TRITON_CACHE_DIR=$TMPDIR_LOCAL/.cache/triton/.cache
export TRITON_DUMP_DIR=$TMPDIR_LOCAL/.cache/triton/.dump

export HF_HOME=$TMPDIR_LOCAL/.cache/huggingface
export HF_DATASETS_CACHE=$TMPDIR_LOCAL/.cache/huggingface/datasets

export WANDB_DIR=$TMPDIR_LOCAL/wandb
export WANDB_CACHE_DIR=$TMPDIR_LOCAL/wandb/cache
export WANDB_CONFIG_DIR=$TMPDIR_LOCAL/wandb/config


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

echo "LS everything!"
echo "LS TMPDIR_LUSTRE!"
ls $TMPDIR_LUSTRE

echo "LS TMPDIR_LOCAL!"
ls $TMPDIR_LOCAL

node=$( scontrol show hostnames $SLURM_JOB_NODELIST )
node_array=($(echo "$node" | tr '\n' '\n'))
printf "%s slots=$NUM_GPUS_PER_NODE\n" "${{node_array[@]}}" > {config["folder_full_path"]}/hostfile
cat {config["folder_full_path"]}/hostfile



srun apptainer exec --nv --no-home \\
  --mount type=bind,src=$TMPDIR_LOCAL,dst=$TMPDIR_LOCAL \\
  --mount type=bind,src=$TMPDIR_LUSTRE,dst=$TMPDIR_LUSTRE \\
  --bind {config["folder_full_path"]}:$TMPDIR_LUSTRE/mount \\
  {config["image_name"]} \\
  bash -c "



# Add user local bin to PATH to avoid warnings
export PATH=\\$HOME/.local/bin:\\$PATH

# Fix CUDA library linking
ln -sf /.singularity.d/libs /usr/local/cuda/compat/lib 2>/dev/null || true



# Set up library paths
export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12/lib64
cd \\$TMPDIR_LUSTRE/mount
export PYTHONPATH=.
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

echo \\"MASTER_ADDR: $MASTER_ADDR\\"
echo \\"MASTER_PORT: $MASTER_PORT\\"
echo \\"NUM_NODES: $NUM_NODES\\"
echo \\"NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE\\"
echo \\"SLURM_PROCID: \\$SLURM_PROCID\\"
echo \\"SLURM_NODEID: \\$SLURM_NODEID\\"

PYTHONUNBUFFERED=1 \\
PYTHONPATH=. \\
python3 -m torch.distributed.run \\
  --nproc_per_node $NUM_GPUS_PER_NODE --nnodes $NUM_NODES --node_rank \\$SLURM_NODEID \\
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \\
  src/training.py \\
  --model_name {config["run_config"]["model_name"]} \\
  --tokenizer_name {config["run_config"]["tokenizer_name"]} \\
  --dataset_name {config["run_config"]["dataset_name"]} \\
  --job_id $SLURM_JOB_ID

"

echo "END TIME: $(date)"
"""
print()
slurm_file = Path(config["slurm_config"]["slurm_file_name"])
slurm_file.write_text(slurm_template)

# Optionally submit automatically
import subprocess

subprocess.run(["sbatch", str(slurm_file)])
