#!/usr/bin/env python3
"""
launch_swarm_experiment.py - Automated Swarm Experiment Launcher

Launches Unity with Swarm scene and the swarm_pointcloud_builder.py script.
Both processes communicate via file-based captures for autonomous reconstruction.
"""

import subprocess
import time
import sys
import os
import shutil
from pathlib import Path

# Check and create D:/ drive directories if they don't exist
D_DRIVE_BASE = r"D:\advaith\unity-run-files"
if os.path.exists("D:\\"):
    os.makedirs(os.path.join(D_DRIVE_BASE, "FinalPointClouds"), exist_ok=True)
    os.makedirs(os.path.join(D_DRIVE_BASE, "ProcessedImages", "SwarmPointClouds"), exist_ok=True)
    os.makedirs(os.path.join(D_DRIVE_BASE, "ProcessedImages", "SwarmCapture"), exist_ok=True)
    print(f"D:/ drive directories ready at {D_DRIVE_BASE}")

def launch_python_only(
    project_path: str = r"C:\Users\sriram\vr_swarm_simulation",
    unity_scene: str = r"Assets\Scenes\Swarm_House2.unity",
    python_script: str = r"Assets\Scripts\swarm\swarm_pointcloud_builder.py",
    python_venv: str = r"vrswarm_env\Scripts\python.exe",
    num_drones: int = 2,
    max_interval: int = 5,
    cleanup: bool = True
):
    """
    Launch Python swarm script and Unity (assumes Unity already running).
    Unity captures at 1Hz, then we post-process for multiple intervals.
    Also triggers Unity to enter Play mode automatically.
    
    Args:
        project_path: Path to Unity project root
        unity_scene: Relative path to Unity scene
        python_script: Relative path to Python swarm script
        python_venv: Path to Python executable in venv
        num_drones: Number of drones in swarm
        max_interval: Maximum capture interval - will process intervals 1 through max_interval
        cleanup: Whether to delete capture folders after processing (default: True)
    """
    
    print("=" * 70)
    print("          SWARM EXPERIMENT LAUNCHER (Auto-Play)")
    print("=" * 70)
    print(f"\nProject:     {project_path}")
    print(f"Unity Scene: {unity_scene}")
    print(f"Python Script: {python_script}")
    print("=" * 70)
    
    # Construct full paths
    script_path = os.path.join(project_path, python_script)
    python_exe = os.path.join(project_path, python_venv)
    trigger_file = os.path.join(project_path, "swarm_play_trigger.txt")
    done_file = os.path.join(project_path, "swarm_done.txt")
    config_file = os.path.join(project_path, "swarm_config.json")
    final_pc_folder = r"D:\advaith\unity-run-files\FinalPointClouds"
    swarm_pc_folder = r"D:\advaith\unity-run-files\ProcessedImages\SwarmPointClouds"
    swarm_capture_folder = r"D:\advaith\unity-run-files\ProcessedImages\SwarmCapture"
    
    # Clean up any leftover done file from previous run
    if os.path.exists(done_file):
        os.remove(done_file)
    
    # Create experiment configuration
    import json
    config = {
        "numDrones": num_drones,
        "dronesAlongX": num_drones,
        "dronesAlongZ": 1,
        "captureInterval": 1.0  # Always capture at 1Hz for multi-interval processing
    }
    
    print(f"\n{'='*70}")
    print("Writing experiment configuration...")
    print("=" * 70)
    print(f"  Drones: {num_drones} ({num_drones}x1 grid)")
    print(f"  Unity capture: 1Hz (1 second intervals)")
    print(f"  Post-process intervals: 1 through {max_interval}s")
    print(f"  Auto-stop when converged (max 300s timeout)")
    print(f"  Cleanup captures: {cleanup}")
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  Config saved to {config_file}")
    
    # Verify files exist
    if not os.path.exists(script_path):
        print(f"\n ERROR: Python script not found: {script_path}")
        return
    
    if not os.path.exists(python_exe):
        print(f"\n⚠ WARNING: Python venv not found at {python_exe}")
        print("  Using system Python instead")
        python_exe = sys.executable
    
    # Trigger Unity to enter Play mode
    print(f"\n{'='*70}")
    print("Triggering Unity Play mode...")
    print("=" * 70)
    print("  Writing trigger file for Unity...")
    
    with open(trigger_file, 'w') as f:
        f.write("play")
    
    print("  Trigger sent - Unity should enter Play mode automatically")
    print(f"  (Make sure Unity editor is open with '{unity_scene}' loaded)")
    
    # Wait a moment for Unity to respond
    print("\n  Waiting 30 seconds for Unity to start playing...")
    time.sleep(30)
    
    script_dir = os.path.dirname(script_path)
    
    print(f"\n{'='*70}")
    print("PHASE 1: Unity Capture at 1Hz")
    print("=" * 70)
    print("  Waiting for Unity to converge and create swarm_done.txt...")
    print("  (Unity is capturing at 1Hz in the background)")
    print("=" * 70)
    
    # Wait for Unity to complete (indicated by swarm_done.txt)
    start_time = time.time()
    timeout = 520  # 500s max experiment + 20s buffer
    
    try:
        while True:
            if os.path.exists(done_file):
                print(f"\n Unity convergence detected (swarm_done.txt exists)")
                break
            
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"\n⚠ Timeout after {elapsed:.0f}s - Unity may not have completed")
                return
            
            time.sleep(1)
        
        # Read convergence time from done file
        convergence_time = None
        if os.path.exists(done_file):
            try:
                with open(done_file, 'r') as f:
                    content = f.read().strip()
                if content.startswith('done,'):
                    convergence_time = float(content.split(',')[1])
                    print(f" Swarm converged in {convergence_time:.2f}s")
                elif content.startswith('timeout,'):
                    print(f"⚠ Experiment reached timeout")
            except Exception as e:
                print(f"  Warning: Could not parse convergence time: {e}")
        
        print(f"\n{'='*70}")
        print(f"PHASE 2: Post-Processing at Multiple Intervals (1-{max_interval})")
        print("=" * 70)
        
        os.makedirs(final_pc_folder, exist_ok=True)
        
        # Process each interval
        for interval in range(1, max_interval + 1):
            print(f"\n--- Processing Interval {interval}s ---")
            
            # Run swarm_pointcloud_builder.py in batch mode with specific interval
            try:
                result = subprocess.run(
                    [
                        python_exe,
                        os.path.basename(script_path),
                        "--batch-mode",
                        "--capture-interval", str(interval),
                        "--capture-dir", swarm_capture_folder,
                        "--output-dir", swarm_pc_folder
                    ],
                    cwd=script_dir,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"   Processing complete for interval {interval}s")
                else:
                    print(f"   Processing failed for interval {interval}s (exit code: {result.returncode})")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}")
            except Exception as e:
                print(f"   Error processing interval {interval}s: {e}")
                continue
            
            # Find and rename the generated point cloud
            if os.path.exists(swarm_pc_folder):
                raw_files = sorted([f for f in os.listdir(swarm_pc_folder) if f.startswith('pointcloud_raw_')])
                if raw_files:
                    latest_raw = raw_files[-1]
                    src = os.path.join(swarm_pc_folder, latest_raw)
                    
                    # New naming format: swarm_raw_{num_drones}_drones_{interval}_interval_{time}s.ply
                    if convergence_time is not None:
                        dst_name = f"swarm_raw_{num_drones}_drones_{interval}_interval_{convergence_time:.2f}s.ply"
                    else:
                        dst_name = f"swarm_raw_{num_drones}_drones_{interval}_interval.ply"
                    dst = os.path.join(final_pc_folder, dst_name)
                    
                    shutil.copy2(src, dst)
                    print(f"   Saved: {dst_name}")
                    
                    # Clean up intermediate point cloud
                    try:
                        os.remove(src)
                    except:
                        pass
        
        print(f"\n{'='*70}")
        print(f"All {max_interval} intervals processed successfully")
        print("=" * 70)
        
        # Conditional cleanup of captures and done file
        if cleanup:
            print("\n" + "="*70)
            print("Cleaning up SwarmCapture folder...")
            print("="*70)
            
            try:
                from send2trash import send2trash
                use_recycle_bin = True
            except ImportError:
                print("  Warning: send2trash not installed, using permanent delete")
                use_recycle_bin = False
            
            if os.path.exists(swarm_capture_folder):
                deleted_items = 0
                for item in os.listdir(swarm_capture_folder):
                    item_path = os.path.join(swarm_capture_folder, item)
                    try:
                        if use_recycle_bin:
                            send2trash(item_path)
                        else:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                            else:
                                shutil.rmtree(item_path)
                        deleted_items += 1
                    except Exception as e:
                        print(f"  Warning: Could not delete {item}: {e}")
                
                print(f"   Cleaned up {deleted_items} items from SwarmCapture")
            else:
                print("   SwarmCapture folder not found")
        else:
            print(f"\n  Skipping cleanup (--no-cleanup flag set)")
        
        # Clean up done file after reading
        if os.path.exists(done_file):
            os.remove(done_file)
        
        print("\nExperiment complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        
    print("\n" + "="*70)
    print("Launcher finished")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch automated swarm reconstruction experiment"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=r"Assets\Scenes\Swarm_House5.unity",
        help="Path to Unity scene (default: Assets\\Scenes\\Swarm_Trial.unity)"
    )
    parser.add_argument(
        "--num-drones",
        type=int,
        default=2,
        help="Number of drones in swarm (default: 2)"
    )
    parser.add_argument(
        "--max-interval",
        type=int,
        default=5,
        help="Maximum capture interval - will process 1 through max (default: 5)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep capture folders after processing (default: cleanup enabled)"
    )
    
    args = parser.parse_args()
    
    launch_python_only(
        unity_scene=args.scene,
        num_drones=args.num_drones,
        max_interval=args.max_interval,
        cleanup=not args.no_cleanup
    )
