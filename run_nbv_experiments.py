#!/usr/bin/env python3
"""
Automated NBV Experiment Runner
Runs launch_nbv_experiment.py multiple times with different drone counts and iteration counts.
"""

"""TO RUN:
# default run (1-5 drones, 0-5 iterations)
python run_nbv_experiments.py --max-drones 5 --max-iterations 5 --skip-existing --delay 5

# Custom run (3-4 drones, 3-4 iterations)
python run_nbv_experiments.py --min-drones 3 --max-drones 4 --min-iterations 3 --max-iterations 4 --skip-existing --delay 5

# Longer timeout for complex experiments (1200 seconds = 20 minutes)
python run_nbv_experiments.py --max-drones 5 --max-iterations 5 --timeout 1200 --skip-existing --delay 5
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


def run_experiment(num_drones, num_iterations, skip_unity=True, timeout=600):
    """Run a single NBV experiment with specified parameters.
    
    Args:
        num_drones: Number of drones to spawn
        num_iterations: Number of NBV iterations to perform
        skip_unity: Whether to skip Unity launch (use existing Unity instance)
        timeout: Maximum time in seconds to wait for experiment (default: 600)
    
    Returns:
        True if experiment succeeded, False otherwise
    """
    cmd = [
        sys.executable,  # Use current Python interpreter
        "launch_nbv_experiment.py",
        "--num-drones", str(num_drones),
        "--max-iterations", str(num_iterations)
    ]
    
    if skip_unity:
        cmd.append("--skip-unity")
    
    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENT: {num_drones} drones, {num_iterations} iterations")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {timeout} seconds")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, check=True, timeout=timeout)
        
        print(f"\n{'='*70}")
        print(f"✓ COMPLETED: {num_drones} drones, {num_iterations} iterations")
        print(f"{'='*70}\n")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n{'='*70}")
        print(f"✗ TIMEOUT: {num_drones} drones, {num_iterations} iterations")
        print(f"Experiment exceeded {timeout} seconds and was terminated")
        print(f"{'='*70}\n")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*70}")
        print(f"✗ FAILED: {num_drones} drones, {num_iterations} iterations")
        print(f"Error: {e}")
        print(f"{'='*70}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"⚠ INTERRUPTED by user")
        print(f"{'='*70}\n")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple NBV experiments with varying drone counts and iterations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments with 1-3 drones and 0-2 iterations (9 total experiments)
  python run_nbv_experiments.py --max-drones 3 --max-iterations 2
  
  # Start with 2 drones, go up to 5 drones, with 0-3 iterations
  python run_nbv_experiments.py --min-drones 2 --max-drones 5 --max-iterations 3
  
  # Skip experiments that already have saved results
  python run_nbv_experiments.py --max-drones 3 --max-iterations 2 --skip-existing
  
  # Add delay between experiments (useful for Unity to settle)
  python run_nbv_experiments.py --max-drones 3 --max-iterations 2 --delay 5
        """
    )
    
    parser.add_argument(
        "--min-drones",
        type=int,
        default=1,
        help="Minimum number of drones (default: 1)"
    )
    
    parser.add_argument(
        "--max-drones",
        type=int,
        required=True,
        help="Maximum number of drones (required)"
    )
    
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=0,
        help="Minimum number of iterations (default: 0)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        required=True,
        help="Maximum number of iterations (required)"
    )
    
    parser.add_argument(
        "--skip-unity",
        action="store_true",
        default=True,
        help="Skip Unity launch (use existing Unity instance) - default: True"
    )
    
    parser.add_argument(
        "--no-skip-unity",
        action="store_true",
        help="Launch Unity for each experiment (slower)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have results in FinalPointClouds"
    )
    
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Delay in seconds between experiments (default: 2)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum time in seconds per experiment before timeout (default: 600)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_drones < 1:
        print("Error: --min-drones must be at least 1")
        return 1
    
    if args.max_drones < args.min_drones:
        print("Error: --max-drones must be >= --min-drones")
        return 1
    
    if args.min_iterations < 0:
        print("Error: --min-iterations must be >= 0")
        return 1
    
    if args.max_iterations < args.min_iterations:
        print("Error: --max-iterations must be >= --min-iterations")
        return 1
    
    # Determine skip_unity setting
    skip_unity = not args.no_skip_unity if args.no_skip_unity else args.skip_unity
    
    # Calculate total experiments
    num_drone_configs = args.max_drones - args.min_drones + 1
    num_iteration_configs = args.max_iterations - args.min_iterations + 1
    total_experiments = num_drone_configs * num_iteration_configs
    
    # Print experiment plan
    print(f"\n{'='*70}")
    print(f"NBV EXPERIMENT BATCH")
    print(f"{'='*70}")
    print(f"Drone range: {args.min_drones} to {args.max_drones} ({num_drone_configs} configs)")
    print(f"Iteration range: {args.min_iterations} to {args.max_iterations} ({num_iteration_configs} configs)")
    print(f"Total experiments: {total_experiments}")
    print(f"Skip Unity launch: {skip_unity}")
    print(f"Skip existing results: {args.skip_existing}")
    print(f"Delay between experiments: {args.delay}s")
    print(f"Timeout per experiment: {args.timeout}s")
    print(f"{'='*70}\n")
    
    # Check for existing results if requested
    final_pc_folder = Path("Assets/FinalPointClouds")
    
    # Run experiments
    completed = 0
    failed = 0
    skipped = 0
    experiment_num = 0
    
    try:
        for num_drones in range(args.min_drones, args.max_drones + 1):
            for num_iterations in range(args.min_iterations, args.max_iterations + 1):
                experiment_num += 1
                
                # Check if results already exist
                if args.skip_existing:
                    expected_file = final_pc_folder / f"raw_{num_drones}_drones_{num_iterations}_iterations.ply"
                    if expected_file.exists():
                        print(f"\n[{experiment_num}/{total_experiments}] Skipping {num_drones} drones, {num_iterations} iterations (result exists)")
                        skipped += 1
                        continue
                
                print(f"\n[{experiment_num}/{total_experiments}] Running experiment...")
                
                success = run_experiment(num_drones, num_iterations, skip_unity, timeout=args.timeout)
                
                if success:
                    completed += 1
                else:
                    failed += 1
                
                # Wait between experiments (except after last one)
                if experiment_num < total_experiments:
                    print(f"Waiting {args.delay} seconds before next experiment...\n")
                    time.sleep(args.delay)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"BATCH COMPLETE")
        print(f"{'='*70}")
        print(f"Total experiments planned: {total_experiments}")
        print(f"Completed successfully: {completed}")
        print(f"Failed: {failed}")
        if args.skip_existing:
            print(f"Skipped (already exist): {skipped}")
        print(f"{'='*70}\n")
        
        return 0 if failed == 0 else 1
        
    except KeyboardInterrupt:
        print(f"\n{'='*70}")
        print(f"BATCH INTERRUPTED")
        print(f"{'='*70}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        if args.skip_existing:
            print(f"Skipped: {skipped}")
        print(f"Remaining: {total_experiments - experiment_num}")
        print(f"{'='*70}\n")
        return 130  # Standard exit code for SIGINT


if __name__ == "__main__":
    sys.exit(main())
