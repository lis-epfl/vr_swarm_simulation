# Quick Start Guide - Swarm Image Capture

## 🚀 Quick Setup (5 minutes)

### Unity Setup
1. Create empty GameObject named "SwarmImageCapture"
2. Add component: `SwarmImageCapture.cs`
3. Set SwarmManager mode to **OLFATI_SABER**
4. Press Play

### Python Setup
```bash
cd /Users/advaithsriram/vr_swarm_simulation/Assets/Scripts/swarm
python swarm_pointcloud_builder.py
```

That's it! Point clouds will be saved to:
- `Assets/ProcessedImages/SwarmPointClouds/`

---

## 📁 File Locations

| File | Path |
|------|------|
| Unity Script | [SwarmImageCapture.cs](file:///Users/advaithsriram/vr_swarm_simulation/Assets/Scripts/swarm/SwarmImageCapture.cs) |
| Python Script | [swarm_pointcloud_builder.py](file:///Users/advaithsriram/vr_swarm_simulation/Assets/Scripts/swarm/swarm_pointcloud_builder.py) |
| Captures | `Assets/ProcessedImages/SwarmCapture/` |
| Point Clouds | `Assets/ProcessedImages/SwarmPointClouds/` |

---

## ⚙️ Key Settings

### Unity (SwarmImageCapture component)
- **Capture Interval**: 5 seconds (adjustable)
- **Resolution**: 300x300 (matches NBV)
- **Mode Check**: Only runs in OLFATI_SABER mode

### Python (command line args)
```bash
# Default
python swarm_pointcloud_builder.py

# Custom settings
python swarm_pointcloud_builder.py \
  --capture-dir /path/to/captures \
  --output-dir /path/to/output \
  --poll-interval 1.0 \
  --sam-model vit_h
```

---

## 🔍 Verify It's Working

### Unity Console
```
[SwarmImageCapture] Initialized with 3 drones
[SwarmImageCapture] Starting capture 0 at 20251202_122534
[SwarmImageCapture]   Drone 0: RGB, Depth, Pose saved
[SwarmImageCapture]   Drone 1: RGB, Depth, Pose saved
[SwarmImageCapture]   Drone 2: RGB, Depth, Pose saved
[SwarmImageCapture] Capture complete: capture_20251202_122534
```

### Python Output
```
Processing: capture_20251202_122534
  Drone count: 3
  Total fused points: 24227
✓ Processing complete!
  Raw: 24227 points
  Downsampled: 16000 points
```

---

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| No captures | Check SwarmManager mode is OLFATI_SABER |
| Python not detecting | Check paths match in both Unity and Python |
| Empty point clouds | Verify DroneDepthCamera component on drones |
| No segmentation | Ensure drones are pointing at objects |

---

## 📊 Output Files

### Every 5 seconds, Unity creates:
```
capture_YYYYMMDD_HHMMSS/
├── metadata.json
├── drone_0_rgb.png
├── drone_0_depth.raw
├── drone_0_pose.json
└── ...
```

### Python processes and creates:
```
pointcloud_raw_YYYYMMDD_HHMMSS.ply         (full merged cloud)
pointcloud_downsampled_YYYYMMDD_HHMMSS.ply (16,000 points)
```

---

## 🎯 Next Steps

1. **Test**: Run with 2-3 drones first
2. **Visualize**: Open PLY files in CloudCompare
3. **Tune**: Adjust capture interval if needed
4. **Scale**: Test with full swarm

For detailed documentation, see [walkthrough.md](file:///Users/advaithsriram/.gemini/antigravity/brain/6a4c2d6f-38b5-467c-978e-8a78277b04c9/walkthrough.md)
