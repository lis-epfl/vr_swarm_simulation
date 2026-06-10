# VR Swarm Drone Racing Course System

Complete ordered ring gate racing system with visual feedback, timing, and navigation guidance.

## Quick Start

1. **Add RingGateVisual** to each gate: Add Component → `RingGateVisual`
2. **Create CourseManager**: GameObject with `RingGateManager` component
   - Populate `gates` list with your gates in order
   - (Optional) Assign `CourseTimer` for timing display
3. **Create CourseStart**: Trigger GameObject with `CourseStartTrigger`
   - Set `courseManager` reference
   - Size BoxCollider to span swarm entry width
4. **Create CoursePath**: GameObject at (0,0,0) with `CoursePathVisual`
   - Set `gateManager` reference
   - Spline path will be auto-generated

**That's it!** No special render pipeline or shader setup needed.

---

## What You Get

### 1. Ordered Gate Progression
- Gates must be cleared in sequence
- Pre-start passes and out-of-order attempts are logged but don't count
- Course completes when all gates cleared

### 2. Visual Feedback
- **Idle ring:** Dim gray
- **Next ring:** Bright cyan + **pulsing scale** (draws attention)
- **Completed ring:** Bright green
- Works with any render pipeline

### 3. Timing System
- Total elapsed time
- Per-gate split times
- Optional on-screen display via TextMeshProUGUI

### 4. Navigation Spline
- Cyan path through all gate centers
- Animated dots flowing toward next gate
- Automatically generated from gate positions

---

## Files

### New Components
- **RingGateVisual.cs** — Color + scale feedback on each gate
- **CourseTimer.cs** — Timing tracker (total + splits)
- **CourseStartTrigger.cs** — Start line trigger
- **CoursePathVisual.cs** — Spline path renderer + animated dots

### Modified
- **RingGateManager.cs** — Added ordered course state management

### Documentation
- **COURSE_SETUP_GUIDE.md** — Complete scene setup walkthrough
- **UPDATED_VISUAL_FEEDBACK.md** — Visual design explanation
- **QUICK_BLOOM_FIX.md** — (Legacy) For optional HDR bloom effects

---

## Inspector Settings Reference

### RingGateVisual (per gate)
```
idleColor:        (0.5, 0.5, 0.5)     # dim gray
nextColor:        (0, 1, 1)           # bright cyan
completedColor:   (0, 1, 0)           # bright green
pulseFrequency:   1.2                 # Hz
pulseAmount:      0.1                 # ±10% scale
baseScale:        1.0
```

### CourseTimer (optional)
```
timerDisplay:     TextMeshProUGUI     # UI text element
displayFormat:    "{0:mm\\:ss\\.ff}"  # MM:SS.FF
```

### CourseStartTrigger
```
courseManager:    RingGateManager     # reference
droneTag:         "Player"
singleFirePerSession: true
```

### CoursePathVisual
```
gateManager:      RingGateManager     # reference
splineSampleCount: 200
lineWidth:        0.15
pathColor:        (0, 1, 1, 0.7)      # cyan with transparency
dotCount:         12
dotSpeed:         5
dotColor:         (0, 0.8, 1)
```

---

## Event Hooks

All events exposed via Inspector on RingGateManager:

- **onNextGateChanged(int gateIndex, RingGate gate)** — When active gate advances
- **onGateCleared(int gateIndex, RingGate gate)** — When a gate completes
- **onRunComplete(CourseRunSummary summary)** — When course finishes

Connect these to UI, audio, analytics, or custom logic.

---

## API Reference

### RingGateManager
```csharp
void StartCourse()               // Begin course
void ResetAll()                 // Reset for new attempt
bool IsCourseRunning { get; }  // Is a run in progress?
int CurrentGateIndex { get; }  // Which gate is next? (-1 = not started)
Transform[] GetCenterPoints()  // All gate centers (for spline)
float GetOverallAccuracy()     // Accuracy % across all gates
```

### CourseTimer
```csharp
void StartTimer()              // Start timing
void RecordGateSplit()        // Record gate completion time
void StopTimer()              // Freeze timer
void Reset()                  // Reset to zero
float TotalElapsed { get; }   // Total seconds
List<float> GateSplits { get; } // Per-gate splits
bool IsRunning { get; }       // Is timer running?
```

### RingGateVisual
```csharp
void SetState(GateVisualState state)  // Idle / Next / Completed
GateVisualState CurrentState { get; }
```

---

## Customization

### Change Gate Colors
Select gate's RingGateVisual in Inspector:
- `idleColor` — Default gate appearance
- `nextColor` — Active target color
- `completedColor` — Completed gate color

### Adjust Pulse Speed
- `pulseFrequency` — Oscillations per second (1.2 Hz default)
- `pulseAmount` — Scale variation (±10% default)

### Change Spline Path Appearance
Select CoursePath's CoursePathVisual:
- `pathColor` — Line color + transparency
- `dotColor` — Dot color
- `pulseAmplitude` — Opacity variation

---

## Troubleshooting

**Gates not rendering:**
- Check Quality Settings → Custom Render Pipeline is **(none)**
- Verify gates are in the `gates` list on CourseManager

**No color changes in Play mode:**
- Gates should start dim gray (Idle)
- Move swarm through CourseStart trigger
- First gate should turn bright cyan (Next)
- Confirm gate materials are rendering (check scene view)

**Spline path not visible:**
- CoursePath must be at world origin (0, 0, 0)
- Verify `gateManager` reference is assigned
- Check all gates have centerPoint children

**Timer not displaying:**
- Assign a TextMeshProUGUI element to `CourseTimer.timerDisplay`
- Verify text element is visible on screen

---

## Notes

- **Render Pipeline:** Works with Default, URP, or HDRP
- **No bloom/HDR required:** Visual feedback uses color + scale
- **Scalable:** Supports any number of gates
- **Customizable:** All colors, timings, and speeds adjustable in Inspector

---

## Example Scene Setup

```
Scene Root
├── GameManager (existing swarm manager)
├── CourseManager
│   └── RingGateManager (with gates list + timer)
├── CourseStart
│   └── BoxCollider (trigger, invisible)
│   └── CourseStartTrigger (→ courseManager ref)
├── Gate_0, Gate_1, Gate_2, ...
│   ├── RingGate
│   ├── RingGateVisual ← NEW
│   ├── RingMeshGenerator
│   └── MeshRenderer
└── CoursePath (at 0,0,0)
    └── CoursePathVisual (→ gateManager ref)
```

---

For detailed setup instructions, see **COURSE_SETUP_GUIDE.md**
