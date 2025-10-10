# NBV Vision System Setup Guide

## 🎯 Overview

Your vision system is now complete! Here's what we've built:

```
Unity (C#)                      Python
├── NBV.cs                     ├── NBVImageProcessor.py
├── NBVImageCapture.cs ←→ Shared Memory ←→ (OpenCV Processing)
└── NBVImageIntegration.cs     └── (Position Commands)
```

## 🚀 Quick Start

### **Step 1: Unity Setup**

**For dynamically spawned drones, create a dedicated Vision Manager:**

1. **Create a new empty GameObject** in your scene hierarchy:
   ```
   Right-click in Hierarchy → Create Empty → Name it "VisionManager"
   ```

2. **Add NBVImageCapture.cs** to the VisionManager:
   ```
   Select VisionManager → Add Component → NBVImageCapture
   ```

3. **Add NBVImageIntegration.cs** to the VisionManager:
   ```
   Select VisionManager → Add Component → NBVImageIntegration
   ```

**Alternative: Add to existing SwarmManager GameObject**
   ```
   Select SwarmManager → Add Components → NBVImageCapture + NBVImageIntegration
   ```

4. **Configure NBVImageCapture settings:**
   - Image Width: 300
   - Image Height: 300  
   - Capture Interval: 0.5 (captures every 0.5 seconds)
   - Max Drone Count: 10
   - Enable Debug Logging: ✓ (for testing)

5. **Configure NBVImageIntegration settings:**
   - Enable Vision Control: ✓
   - Vision Influence Strength: 0.5 (50% influence)
   - Show Vision Influence: ✓ (for debugging)

**Note:** The components will automatically find and work with drones spawned at runtime!

### **Step 2: Python Setup**

1. **Activate your virtual environment:**
   ```powershell
   cd "C:\Users\sriram\vr_swarm_simulation"
   .\vrswarm_env\Scripts\activate
   ```

2. **Install required packages** (if not already done):
   ```powershell
   pip install opencv-python numpy
   ```

3. **Run the image processor:**
   ```powershell
   cd "Assets\Scripts\swarm"
   python NBVImageProcessor.py
   ```

### **Step 3: Test the System**

1. **Start Unity** with your scene running
2. **Start the Python processor** 
3. **Watch the console logs** in both Unity and Python
4. **Enable debug visualization** to see vision influence

## 📊 Expected Behavior

### **Unity Console:**
```
NBVImageCapture initialized with 3 drone cameras
Sent 3 drone images to shared memory
Vision command received: (0.1, 0.0, 0.0)
Applying vision influence: Offset=(0.05, 0.0, 0.0)
```

### **Python Console:**
```
✅ NBVImageProcessor initialized successfully
🖼️ Drone 0: Brightness=120.5, Edges=1250
🧠 Analysis: Brightness=115.2, EdgeDensity=0.085
🎯 Command: Move (0.10, 0.00, 0.00)
📤 Sent position command: (0.10, 0.00, 0.00)
📊 Processed 10 image batches, sent 8 commands
```

### **Visual Debug:**
- **Blue sphere**: Original NBV center point
- **Green sphere**: Current NBV center point (with vision influence)
- **Red line**: Vision influence vector
- **Magenta arrow**: Current vision command direction

## 🔧 Configuration Options

### **NBVImageCapture.cs Settings:**

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| Image Width/Height | Resolution of captured images | 300x300 |
| Capture Interval | How often to capture (seconds) | 0.5 |
| Max Drone Count | Maximum drones to capture | 10 |
| Debug Logging | Enable console logging | ✓ (for testing) |

### **NBVImageIntegration.cs Settings:**

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| Vision Influence Strength | How much vision affects movement (0-1) | 0.5 |
| Vision Command Timeout | Timeout for vision commands (seconds) | 5.0 |
| Smooth Vision Commands | Apply smoothing to commands | ✓ |
| Smoothing Speed | Speed of smoothing | 2.0 |

### **NBVImageProcessor.py Settings:**

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| processing_interval | How often to process (seconds) | 2.0 |
| debug | Enable debug logging | True |

## 🐛 Troubleshooting

### **Problem: Python can't connect to shared memory**
```
❌ Failed to initialize shared memory: [Errno 2] No such file or directory
```
**Solution:** Make sure Unity is running and NBVImageCapture is active.

### **Problem: No images being received**
```
⚠️ No new images available
```
**Solution:** 
- Check that drones have "DroneBase" tag
- Verify drones have "FPV" child with Camera component
- Enable debug logging in NBVImageCapture

### **Problem: Vision commands not affecting drones**
**Solution:**
- Check that NBVImageIntegration is on same GameObject as NBV.cs
- Verify "Enable Vision Control" is checked
- Increase "Vision Influence Strength"

### **Problem: Drones moving erratically**
**Solution:**
- Reduce "Vision Influence Strength" (try 0.1-0.3)
- Enable "Smooth Vision Commands"
- Increase "Smoothing Speed"

## 🔄 System Flow

1. **NBVImageCapture.cs** captures drone images every 0.5s
2. **Images written** to shared memory (`NBVImageSharedMemory`)
3. **NBVImageProcessor.py** reads images from shared memory
4. **Python processes** images using OpenCV (placeholder logic)
5. **Position commands** calculated based on image analysis
6. **Commands written** to shared memory (`NBVCommandSharedMemory`)
7. **NBVImageIntegration.cs** reads position commands
8. **Commands applied** to NBV center point with configurable influence
9. **NBV.cs** moves drones using modified center point

## 🎮 Testing Commands

### **Python Testing:**
```python
# Test individual components
from NBVImageProcessor import NBVImageProcessor

processor = NBVImageProcessor(debug=True)
if processor.start():
    print("✅ System working!")
else:
    print("❌ Check Unity connection")
```

### **Unity Testing:**
```csharp
// Add this to a test script to verify integration
var integration = FindObjectOfType<NBVImageIntegration>();
Debug.Log($"Vision active: {integration.IsVisionActive()}");
Debug.Log($"Current command: {integration.GetCurrentVisionCommand()}");
```

## 🚀 Next Steps

### **Phase 1: Basic Testing** ✅
- [x] Shared memory communication
- [x] Image capture and transfer
- [x] Position command system
- [x] Integration with NBV.cs

### **Phase 2: Enhanced Processing** (Next)
- [ ] Implement real computer vision algorithms
- [ ] Object detection for obstacle avoidance
- [ ] Feature tracking for navigation
- [ ] Integration with your OpenCV notebook experiments

### **Phase 3: Advanced Features** (Future)
- [ ] Multi-drone coordination through vision
- [ ] Dynamic formation adjustment based on environment
- [ ] Machine learning integration
- [ ] Performance optimization

## 📝 Customization Points

The system is designed to be easily customizable:

1. **NBVImageProcessor.py `process_images()`**: Replace placeholder with your OpenCV algorithms
2. **NBVImageIntegration.cs `ApplyVisionInfluence()`**: Modify how commands affect drone movement
3. **Memory layout**: Adjust shared memory structure for additional data
4. **Processing frequency**: Balance between responsiveness and performance

Your architecture is now ready for real computer vision integration! 🎉