"""Run multiple experiments in batch based on config grid specifications.

Usage:
  uv run scripts/run_batch.py --config configs/main_experiments/main_exp.yaml
Supports parallel execution across multiple nodes with automatic lock-based coordination.
"""

import hashlib
import itertools
import os
import subprocess
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import yaml

from src.experiment.utils import get_logger

logger = get_logger(__name__)

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--skip-completed", action="store_true", default=True)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--ignore-locks", action="store_true")

args = parser.parse_args()


def generate_config_name(grid_values):
    """Generate a descriptive name for a grid configuration."""
    parts = []
    for key, value in grid_values.items():
        # Use short form of the key (last part after dot)
        key_short = key.split(".")[-1]
        # Format value appropriately
        if isinstance(value, (int, float)):
            value_str = str(value)
        else:
            value_str = str(value).replace("_", "")
        parts.append(f"{key_short}_{value_str}")

    # Create hash of full combination for uniqueness
    combo_str = "_".join(f"{k}={v}" for k, v in sorted(grid_values.items()))
    hash_suffix = hashlib.md5(combo_str.encode()).hexdigest()[:6]

    name = "_".join(parts) if parts else "default"
    return f"{name}_{hash_suffix}"


def load_config_with_grid(config_path):
    """Load config and expand any grid specifications."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check if config has grid specification
    if "grid" not in config:
        return [(config, None)]

    # Expand grid into multiple configs
    grid = config.pop("grid")
    # Deep copy to preserve all nested structures
    base_config = yaml.safe_load(yaml.dump(config))
    base_name = base_config["experiment"]["name"]
    base_output_dir = base_config["experiment"]["output_dir"]

    configs = []
    # Generate all combinations from grid

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    for combination in itertools.product(*values):
        new_config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
        grid_values = {}

        for key, value in zip(keys, combination):
            # Navigate nested keys (e.g., "experiment.seed")
            parts = key.split(".")
            target = new_config
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
            grid_values[key] = value

        # Generate unique name for this grid configuration
        config_suffix = generate_config_name(grid_values)

        # Update experiment name and output_dir to include grid parameters
        new_config["experiment"]["name"] = f"{base_name}_{config_suffix}"
        new_config["experiment"]["output_dir"] = f"{base_output_dir.rstrip('/')}/{config_suffix}"

        configs.append((new_config, config_suffix))

    return configs


def run_command(script_name, **kwargs):
    """Run a single experiment command"""
    command = [
        "uv",
        "run",
        f"scripts/{script_name}",
    ]

    # Add appropriate arguments based on script
    if script_name == "run_experiment.py":
        command.extend(["--config_path", kwargs["config_path"]])
        if args.ignore_locks:
            command.append("--ignore-locks")
    elif script_name == "run_eval.py":
        command.extend(["--run_dir", kwargs["run_dir"]])

    logger.warning(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        logger.warning("✅ Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"❌ Failed with exit code {e.returncode}")
        return False


def is_experiment_running(run_dir):
    return ".experiment.lock" in [f.lower() for f in os.listdir(run_dir)]


def is_experiment_completed(run_dir):
    return ".finished" in [f.lower() for f in os.listdir(run_dir)]


def is_evaluation_running(run_dir):
    """Check if evaluation is currently running (lock file exists)."""
    return ".eval.lock" in [f.lower() for f in os.listdir(run_dir)]


def is_evaluation_completed(run_dir):
    """Check if evaluation has already been completed."""
    if "eval" not in [f.lower() for f in os.listdir(run_dir)]:
        return False
    # Check for evaluation results
    eval_files = glob(f"{run_dir}/eval/*.parquet")
    return len(eval_files) > 0


if __name__ == "__main__":
    completed_jobs = 0
    failed_jobs = 0
    skipped_jobs = 0

    if os.path.isdir(args.config):
        config_paths = sorted(glob(f"{args.config}/*.yaml"))
    else:
        config_paths = [args.config]

    warnings = []
    temp_config_files = []  # Track temp files for cleanup

    for config_path in config_paths:
        assert os.path.exists(config_path)

        logger.warning(f"Loading config: {config_path}")

        # Load and expand config with grid if present
        expanded_configs = load_config_with_grid(config_path)
        logger.warning(f"Expanded to {len(expanded_configs)} configuration(s)")

        for i, (config, config_suffix) in enumerate(expanded_configs):
            if config_suffix:
                run_id = f"{Path(config_path).stem}_{config_suffix}"
            else:
                run_id = config["experiment"]["name"]

            logger.warning(f"\n[{i+1}/{len(expanded_configs)}] Processing: {run_id}")

            # Create temporary config file in the output directory
            output_dir = config["experiment"]["output_dir"]
            os.makedirs(output_dir, exist_ok=True)

            temp_config_path = f"{output_dir}/.temp_config_{run_id}.yaml"
            with open(temp_config_path, "w") as f:
                yaml.safe_dump(config, f)
            temp_config_files.append(temp_config_path)

            if args.eval_only:
                # Only evaluate existing runs
                if not is_experiment_completed(output_dir):
                    logger.warning(f"  Skipping {run_id} (experiment not completed)")
                    skipped_jobs += 1
                    continue
                if args.skip_completed and is_evaluation_completed(output_dir):
                    logger.warning(f"  Skipping {run_id} (evaluation already done)")
                    skipped_jobs += 1
                    continue
                if is_evaluation_running(output_dir):
                    logger.warning(
                        f"  Skipping {run_id} (evaluation currently running on another node)"
                    )
                    skipped_jobs += 1
                    continue

                if args.dry_run:
                    logger.warning(
                        f"  [DRY RUN] uv run scripts/run_eval.py --run_dir {output_dir}"
                    )
                    completed_jobs += 1
                else:
                    success = run_command("run_eval.py", run_dir=output_dir)
                    if success:
                        completed_jobs += 1
                    else:
                        failed_jobs += 1
            else:
                # Check if experiment is already completed or running
                if is_experiment_completed(output_dir) and not args.ignore_locks:
                    logger.warning(f"  Skipping {run_id} (experiment already completed)")
                    skipped_jobs += 1
                    # Check if evaluation is needed
                    if not is_evaluation_completed(output_dir) and not is_evaluation_running(
                        output_dir
                    ):
                        logger.warning(f"  Evaluating completed run: {output_dir}")
                        if not args.dry_run:
                            eval_success = run_command("run_eval.py", run_dir=output_dir)
                            if eval_success:
                                completed_jobs += 1
                            else:
                                failed_jobs += 1
                elif is_experiment_running(output_dir) and not args.ignore_locks:
                    logger.warning(
                        f"  Skipping {run_id} (experiment currently running on another node)"
                    )
                    skipped_jobs += 1
                else:
                    # Run new experiment
                    if args.dry_run:
                        logger.warning(
                            f"  [DRY RUN] uv run scripts/run_experiment.py"
                            f" --config_path {temp_config_path}"
                        )
                        completed_jobs += 1
                    else:
                        success = run_command("run_experiment.py", config_path=temp_config_path)
                        if success:
                            completed_jobs += 1
                            # After successful experiment, evaluate if needed
                            if not is_evaluation_completed(
                                output_dir
                            ) and not is_evaluation_running(output_dir):
                                logger.warning(f"  Evaluating new run: {output_dir}")
                                eval_success = run_command("run_eval.py", run_dir=output_dir)
                                if not eval_success:
                                    failed_jobs += 1
                        else:
                            failed_jobs += 1

    logger.warning(f"\n{'='*50}")
    logger.warning("Summary:")
    logger.warning(f"  Completed: {completed_jobs}")
    logger.warning(f"  Failed: {failed_jobs}")
    logger.warning(f"  Skipped: {skipped_jobs}")
    logger.warning(f"{'='*50}")

    if warnings:
        logger.warning("\nWarnings:")
        for w in warnings:
            logger.warning(f"🦺 {w} 🦺")
