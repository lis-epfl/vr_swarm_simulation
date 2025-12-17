#!/usr/bin/env python3
"""
Automated Swarm Experiment Runner
Runs launch_swarm_experiment.py multiple times with different drone counts and capture intervals.
Each experiment saves the final point cloud as swarm_raw_{num_drones}_drones_{interval}_interval.ply
"""

"""TO RUN:
# Run with 1-3 drones, capture intervals 1-5 seconds (15 total experiments)
python run_swarm_experiments.py --max-drones 3 --max-interval 5 --skip-existing --delay 5

# Custom run with 2-4 drones, intervals 2-4 seconds
python run_swarm_experiments.py --min-drones 2 --max-drones 4 --min-interval 2 --max-interval 4 --skip-existing

# With custom timeout (increase if convergence takes longer)
python run_swarm_experiments.py --max-drones 3 --max-interval 3 --timeout 400 --skip-existing
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


def run_experiment(num_drones, capture_interval, timeout=320):
    """Run a single swarm experiment with specified parameters.
    
    Args:
        num_drones: Number of drones in swarm
        capture_interval: Seconds between captures (integer)
        timeout: Maximum time in seconds to wait for experiment (default: 320 = 300s max + 20s buffer)
    
    Returns:
        True if experiment succeeded, False otherwise
    """
    cmd = [
        sys.executable,  # Use current Python interpreter
        "launch_swarm_experiment.py",
        "--num-drones", str(num_drones),
        "--capture-interval", str(capture_interval)
    ]
    
    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENT: {num_drones} drones, {capture_interval}s interval")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Auto-stop on convergence (max 300s), Timeout: {timeout}s")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, check=True, timeout=timeout)
        
        print(f"\n{'='*70}")
        print(f"✓ COMPLETED: {num_drones} drones, {capture_interval}s interval")
        print(f"{'='*70}\n")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n{'='*70}")
        print(f"✗ TIMEOUT: {num_drones} drones, {capture_interval}s interval")
        print(f"Experiment exceeded {timeout} seconds and was terminated")
        print(f"{'='*70}\n")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*70}")
        print(f"✗ FAILED: {num_drones} drones, {capture_interval}s interval")
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
        description="Run multiple swarm experiments with varying drone counts and capture intervals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments with 1-3 drones, capture intervals 1-5s (15 total experiments)
  python run_swarm_experiments.py --max-drones 3 --max-interval 5
  
  # Start with 2 drones, intervals 2-4s
  python run_swarm_experiments.py --min-drones 2 --max-drones 2 --min-interval 2 --max-interval 4
  
  # Skip experiments that already have results
  python run_swarm_experiments.py --max-drones 3 --max-interval 5 --skip-existing
  
  # Add delay between experiments (useful for Unity to settle)
  python run_swarm_experiments.py --max-drones 3 --max-interval 3 --delay 10
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
        "--min-interval",
        type=int,
        default=1,
        help="Minimum capture interval in seconds (default: 1)"
    )
    
    parser.add_argument(
        "--max-interval",
        type=int,
        required=True,
        help="Maximum capture interval in seconds (required)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have results in FinalPointClouds"
    )
    
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Delay in seconds between experiments (default: 5)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=320,
        help="Maximum time in seconds per experiment before timeout (default: 320 = 300s max + 20s buffer)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_drones < 1:
        print("Error: --min-drones must be at least 1")
        return 1
    
    if args.max_drones < args.min_drones:
        print("Error: --max-drones must be >= --min-drones")
        return 1
    
    if args.min_interval < 1:
        print("Error: --min-interval must be at least 1")
        return 1
    
    if args.max_interval < args.min_interval:
        print("Error: --max-interval must be >= --min-interval")
        return 1
    
    # Use timeout from args
    timeout = args.timeout
    
    # Calculate total experiments
    num_drone_configs = args.max_drones - args.min_drones + 1
    num_interval_configs = args.max_interval - args.min_interval + 1
    total_experiments = num_drone_configs * num_interval_configs
    
    # Print experiment plan
    print(f"\n{'='*70}")
    print(f"SWARM EXPERIMENT BATCH")
    print(f"{'='*70}")
    print(f"Drone range: {args.min_drones} to {args.max_drones} ({num_drone_configs} configs)")
    print(f"Interval range: {args.min_interval}s to {args.max_interval}s ({num_interval_configs} configs)")
    print(f"Auto-stop on convergence (max 300s)")
    print(f"Total experiments: {total_experiments}")
    print(f"Skip existing results: {args.skip_existing}")
    print(f"Delay between experiments: {args.delay}s")
    print(f"Timeout per experiment: {timeout}s")
    print(f"{'='*70}\n")
    
    # Check for existing results if requested
    final_pc_folder = Path("Assets/FinalPointClouds")
    
    # Run experiments
    completed = 0
    failed = 0
    skipped = 0
    experiment_num = 0

    start_time = time.time()
    
    try:
        for num_drones in range(args.min_drones, args.max_drones + 1):
            for capture_interval in range(args.min_interval, args.max_interval + 1):
                experiment_num += 1
                
                # Check if result already exists
                if args.skip_existing:
                    expected_file = final_pc_folder / f"swarm_raw_{num_drones}_drones_{capture_interval}_interval.ply"
                    if expected_file.exists():
                        print(f"\n[{experiment_num}/{total_experiments}] Skipping {num_drones} drones, {capture_interval}s interval (exists)")
                        skipped += 1
                        continue
                
                print(f"\n[{experiment_num}/{total_experiments}] Running experiment...")
                
                success = run_experiment(num_drones, capture_interval, timeout=timeout)
                
                if success:
                    completed += 1
                else:
                    failed += 1
                
                # Wait between experiments (except after last one)
                if experiment_num < total_experiments - skipped:
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

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal batch time: {elapsed_time:.2f} seconds for {total_experiments} experiments.\n")
        
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
