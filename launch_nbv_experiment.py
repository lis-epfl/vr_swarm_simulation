#!/usr/bin/env python3
"""
launch_nbv_experiment.py - Automated NBV Experiment Launcher

Launches Unity with NBV-Trial scene and the MAP_NBV_trial.py script.
Both processes communicate via shared memory for autonomous reconstruction.
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def launch_python_only(
    project_path: str = r"C:\Users\sriram\vr_swarm_simulation",
    python_script: str = r"Assets\Scripts\NBV\MAP_NBV_trial.py",
    python_venv: str = r"vrswarm_env\Scripts\python.exe",
    num_drones: int = 2,
    drone_speed: float = 12.0,
    max_iterations: int = 5
):
    """
    Launch only the Python NBV script (assumes Unity already running).
    Also triggers Unity to enter Play mode automatically.
    
    Args:
        project_path: Path to Unity project root
        python_script: Relative path to Python NBV script
        python_venv: Path to Python executable in venv
    """
    
    print("=" * 70)
    print("          NBV EXPERIMENT LAUNCHER (Auto-Play)")
    print("=" * 70)
    print(f"\nProject:     {project_path}")
    print(f"Python NBV:  {python_script}")
    print("=" * 70)
    
    # Construct full paths
    script_path = os.path.join(project_path, python_script)
    python_exe = os.path.join(project_path, python_venv)
    trigger_file = os.path.join(project_path, "nbv_play_trigger.txt")
    config_file = os.path.join(project_path, "nbv_config.json")
    
    # Create experiment configuration
    import json
    config = {
        "numDrones": num_drones,
        "dronesAlongX": num_drones,  # Keep dronesAlongZ = 1 for simplicity
        "dronesAlongZ": 1,
        "droneSpeed": drone_speed,
        "maxIterations": max_iterations
    }
    
    print(f"\n{'='*70}")
    print("Writing experiment configuration...")
    print("=" * 70)
    print(f"  Drones: {num_drones} ({num_drones}x1 grid)")
    print(f"  Speed: {drone_speed}")
    print(f"  Max iterations: {max_iterations}")
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  ✓ Config saved to {config_file}")
    
    # Verify files exist
    if not os.path.exists(script_path):
        print(f"\n❌ ERROR: Python script not found: {script_path}")
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
    
    print("  ✓ Trigger sent - Unity should enter Play mode automatically")
    print("  (Make sure Unity editor is open with NBV-Trial scene loaded)")
    
    # Wait a moment for Unity to respond
    print("\n  Waiting 30 seconds for Unity to start playing...")
    time.sleep(30)
    
    script_dir = os.path.dirname(script_path)
    
    print(f"\n  Script: {python_script}")
    print(f"  Working directory: {script_dir}")
    print(f"  Python: {python_exe}")
    
    try:
        python_process = subprocess.Popen(
            [python_exe, os.path.basename(script_path)],
            cwd=script_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE  # Open in new window
        )
        print(f"  ✓ Python script launched (PID: {python_process.pid})")
    except Exception as e:
        print(f"\n❌ ERROR launching Python script: {e}")
        return
    
    # Monitor process
    print(f"\n{'='*70}")
    print("EXPERIMENT RUNNING")
    print("=" * 70)
    print("  Python NBV script is running in separate window")
    print("  Check Unity for drone movements and vision capture")
    print("\n  Press Ctrl+C here to stop Python script")
    print("=" * 70)
    
    try:
        # Wait for Python script to finish
        returncode = python_process.wait()
        print(f"\n✓ Python script completed (exit code: {returncode})")
        print("\nExperiment complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        print("Stopping Python script...")
        
        try:
            python_process.terminate()
            print("  ✓ Python script stopped")
        except:
            pass
        
    print("\n" + "="*70)
    print("Launcher finished")
    print("="*70)

def find_unity_executable():
    """Find Unity executable in common installation locations."""
    possible_paths = [
        r"C:\Program Files\Unity\Hub\Editor\2022.3.62f2\Editor\Unity.exe",
        # r"C:\Program Files\Unity\Hub\Editor\2022.3.51f1\Editor\Unity.exe",
        r"C:\Program Files\Unity\2022.3.62f2\Editor\Unity.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, ask user
    print("Unity executable not found in default locations.")
    print("Please enter the path to Unity.exe:")
    return input("> ").strip().strip('"')

def launch_nbv_experiment(
    project_path: str = r"C:\Users\sriram\vr_swarm_simulation",
    unity_scene: str = r"Assets\Scenes\NBV-House5.unity",
    python_script: str = r"Assets\Scripts\NBV\MAP_NBV_trial.py",
    wait_for_unity: int = 15,  # seconds to wait for Unity to fully start
    python_venv: str = r"vrswarm_env\Scripts\python.exe"
):
    """
    Launch automated NBV experiment.
    
    Args:
        project_path: Path to Unity project root
        unity_scene: Relative path to Unity scene
        python_script: Relative path to Python NBV script
        wait_for_unity: Seconds to wait for Unity initialization
        python_venv: Path to Python executable in venv
    """
    
    print("=" * 70)
    print("          NBV EXPERIMENT LAUNCHER")
    print("=" * 70)
    print(f"\nProject:     {project_path}")
    print(f"Unity Scene: {unity_scene}")
    print(f"Python NBV:  {python_script}")
    print("=" * 70)
    
    # Find Unity
    unity_exe = find_unity_executable()
    print(f"\nUnity executable: {unity_exe}")
    
    # Construct full paths
    scene_path = os.path.join(project_path, unity_scene)
    script_path = os.path.join(project_path, python_script)
    python_exe = os.path.join(project_path, python_venv)
    
    # Verify files exist
    if not os.path.exists(unity_exe):
        print(f"\n❌ ERROR: Unity executable not found: {unity_exe}")
        return
    
    if not os.path.exists(scene_path):
        print(f"\n❌ ERROR: Unity scene not found: {scene_path}")
        return
    
    if not os.path.exists(script_path):
        print(f"\n❌ ERROR: Python script not found: {script_path}")
        return
    
    if not os.path.exists(python_exe):
        print(f"\n⚠ WARNING: Python venv not found at {python_exe}")
        print("  Using system Python instead")
        python_exe = sys.executable
    
    # Launch Unity
    print(f"\n{'='*70}")
    print("STEP 1: Launching Unity...")
    print("=" * 70)
    
    unity_args = [
        unity_exe,
        "-projectPath", project_path,
        # Open directly to play mode (optional - remove if you want to manually press Play)
        # "-executeMethod", "AutoPlay.Run"  
    ]
    
    print(f"  Opening Unity project: {project_path}")
    print(f"  Scene: {unity_scene}")
    print(f"  You will need to:")
    print(f"    1. Open the scene '{unity_scene}' if not already open")
    print(f"    2. Press PLAY in Unity editor")
    
    try:
        unity_process = subprocess.Popen(
            unity_args,
            cwd=project_path
        )
        print(f"  ✓ Unity launched (PID: {unity_process.pid})")
    except Exception as e:
        print(f"\n❌ ERROR launching Unity: {e}")
        return
    
    # Wait for Unity to initialize
    print(f"\n{'='*70}")
    print(f"STEP 2: Waiting {wait_for_unity} seconds for Unity to initialize...")
    print("=" * 70)
    print("  (Open Unity, load the NBV-Trial scene, and press Play)")
    
    for i in range(wait_for_unity, 0, -1):
        print(f"  Starting NBV script in {i} seconds...", end='\r')
        time.sleep(1)
    print()
    
    # Launch Python NBV script
    print(f"\n{'='*70}")
    print("STEP 3: Launching Python NBV Script...")
    print("=" * 70)
    
    script_dir = os.path.dirname(script_path)
    
    print(f"  Script: {python_script}")
    print(f"  Working directory: {script_dir}")
    print(f"  Python: {python_exe}")
    
    try:
        python_process = subprocess.Popen(
            [python_exe, os.path.basename(script_path)],
            cwd=script_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE  # Open in new window
        )
        print(f"  ✓ Python script launched (PID: {python_process.pid})")
    except Exception as e:
        print(f"\n❌ ERROR launching Python script: {e}")
        unity_process.terminate()
        return
    
    # Monitor processes
    print(f"\n{'='*70}")
    print("EXPERIMENT RUNNING")
    print("=" * 70)
    print("  Unity and Python NBV are now communicating via shared memory")
    print("  Python window: Shows NBV iteration progress and point cloud saves")
    print("  Unity window: Shows drone movements and vision capture")
    print("\n  Press Ctrl+C here to stop both processes")
    print("=" * 70)
    
    try:
        # Wait for Python script to finish
        returncode = python_process.wait()
        print(f"\n✓ Python script completed (exit code: {returncode})")
        
        print("\nExperiment complete! You can now close Unity.")
        print("Keep Unity open if you want to inspect the final drone positions.")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        print("Stopping processes...")
        
        try:
            python_process.terminate()
            print("  ✓ Python script stopped")
        except:
            pass
        
        print("  Unity still running - close manually if needed")
        
    print("\n" + "="*70)
    print("Launcher finished")
    print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch automated NBV reconstruction experiment"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=45,
        help="Seconds to wait for Unity to start (default: 45)"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip waiting, launch Python immediately"
    )
    parser.add_argument(
        "--skip-unity",
        action="store_true",
        help="Skip Unity launch (assume Unity already running with scene loaded)"
    )
    parser.add_argument(
        "--num-drones",
        type=int,
        default=2,
        help="Number of drones to spawn (default: 2)"
    )
    parser.add_argument(
        "--drone-speed",
        type=float,
        default=12.0,
        help="Drone migration speed (default: 12.0)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum NBV iterations (default: 5)"
    )
    
    args = parser.parse_args()
    
    wait_time = 0 if args.no_wait else args.wait
    
    if args.skip_unity:
        # Just launch Python directly
        launch_python_only(
            num_drones=args.num_drones,
            drone_speed=args.drone_speed,
            max_iterations=args.max_iterations
        )
    else:
        launch_nbv_experiment(wait_for_unity=wait_time)
