# CWL Controller Usage Guide

## Overview

The **CWLController** is an adaptive feedback system that automatically adjusts drone control parameters to maintain operator cognitive workload at an optimal (Medium) level. It receives workload classifications from a Python ML model and makes real-time adjustments to flight difficulty.

## Core Concept

The controller maintains a **single step index** (0 to `Num Steps - 1`) that maps to a parameter set across all 6 adjustable drone parameters. This means:
- **Step 0** = minimum difficulty (min profile values)
- **Step Num Steps-1** = maximum difficulty (max profile values)
- All parameters scale together proportionally

## Unity Inspector Parameters

### Feedback System
**CWL Feedback Enabled** (`bool`)
- **Default**: `true`
- **Effect**: Toggle the entire adjustment system on/off
- **Use Case**: Set to `false` to log model recommendations without modifying drone behavior (non-adaptive experiment mode)

### CWL Adjustment Range
These define your difficulty spectrum. All adjustable parameters interpolate between these profiles.

**Min Profile** (FlightProfile)
- **Effect**: Easiest drone behavior (step 0)
- **Typical values**: Slowest speed, lowest responsiveness
- **Setup**: Create a FlightProfile asset or reference an existing one

**Max Profile** (FlightProfile)
- **Effect**: Hardest drone behavior (step Num Steps-1)
- **Typical values**: Fastest speed, highest responsiveness
- **Setup**: Create a FlightProfile asset with challenging parameters

**Default Profile** (FlightProfile)
- **Effect**: Starts with this profile for the first trial
- **Typical setup**: Middle ground between min/max (or use min profile)
- **Usage**: Called when `ResetToDefaultProfile()` is invoked

### Adaptive Algorithm

**Num Steps** (`int`, Min: 1)
- **Default**: `16`
- **Effect**: Total discrete difficulty levels available
- **Range Meaning**: Step index 0→Num Steps-1 represents min→max profile

**Max Step Size** (`int`, Min: 1)
- **Default**: `4`
- **Effect**: Maximum steps to move per inference
- **Limits aggressiveness**: Prevents wild jumps in difficulty

**Warmup Updates** (`int`, Min: 0)
- **Default**: `2`
- **Effect**: Number of inferences to skip before enabling adjustments
- **Purpose**: Lets the system stabilize before CWL feedback kicks in
- **Typical values**: `2-5` (depending on model inference frequency)
- **Activation**: Call `EnableWarmup()` at the start of each trial

**Step Mode** (enum: Linear / Proportional)
- **Default**: `Proportional`

| Mode | Behavior
|------|---------
| **Linear** | Steps increment by 1 per same-direction inference, reset on direction change
| **Proportional** | Steps based on model confidence margin (winner prob − runner-up prob), capped at Max Step Size 

**Enable Sharp Switch Buffer** (`bool`)
- **Default**: `true`
- **Effect**: Prevents immediate High↔Low switches (buffers one inference with 0 steps)
- **Purpose**: Smooths oscillation between very high and very low workload
- **Use Cases**: Might be best if used with the **Linear** mode

## Adjustable Drone Parameters

These 6 parameters are interpolated together across all steps from min to max profile:

| Parameter | Effect 
|-----------|--------
| **maxSpeed** | All drones max velocity
| **maxYawRate** | Rotation speed
| **maxPitch** | Forward/backward tilt 
| **maxRoll** | Left/right banking
| **maxAltitudeRate** | Vertical climb/descent speed
| **maxAlpha** | Control responsiveness/damping

**Important**: All parameters scale proportionally. You cannot adjust individual parameters—they move together as a unified difficulty spectrum.

## API Integration

### CWL Level Endpoint

**Endpoint**: `POST /api/cwl/level`

**Purpose**: Receive ML model inference and confidence scores from external Python process

#### Request Format

```json
{
  "level": "Low|Medium|High",
  "lowProb": 0.1,
  "medProb": 0.2,
  "highProb": 0.7
}
```

#### How Probabilities Affect Tuning (Proportional Mode)

In **Proportional step mode**, the confidence margin determines step size:

```
Margin = max(lowProb, medProb, highProb) − second_highest_prob
Steps = min(Margin * scaling_factor, Max Step Size)
```

| Scenario | Margin | Steps |
|----------|--------|-------|
| Very confident (High=0.85, Med=0.10) | 0.75 | 3-4 (max) |
| Moderately confident (High=0.60, Med=0.30) | 0.30 | 1-2 |
| Uncertain (High=0.35, Med=0.33) | 0.02 | 1 |

**Implication**: More confident predictions drive larger adjustments; uncertain predictions make minimal changes.

### Non-Adaptive Logging (Analysis Mode)
**Use when**: You want to collect data without modifying drone behavior
```
CWL Feedback Enabled: false
(Recommendations logged to console, no adjustments applied)
```

## Step Modes Explained

### Linear Mode Example
```
Inference sequence: Low, Low, Low, High, High, Low
Step deltas:        +1,  +1,  +1,  -1,  -1,  +1
Description:        First Low inference adds 1 step, 
                    second Low adds another 1, etc.
                    Changes direction on High → resets counter
```

### Proportional Mode Example
```
High confidence Low:  margin = 0.70 − 0.20 = 0.50  → 3 steps
Low confidence High:  margin = 0.35 − 0.33 = 0.02  → 1 step
Capped at Max Step Size (e.g., 4)
```

## Warmup Phase Workflow

1. **Trial starts** → Call `EnableWarmup()`
2. **Next N inferences** (where N = Warmup Updates) → Logged but not applied
3. **Inference N+1 onward** → Adjustments begin

**Console output during warmup**:
```
[CWLController] Warmup phase enabled
[CWLController] Warmup phase 1/3
[CWLController] Warmup phase 2/3
[CWLController] Warmup phase 3/3
[CWLController] Warmup phase complete, CWL adjustments will be enabled starting from next inference
```

## Public API for Experiments

### Trial Setup
```csharp
// Enable adaptive feedback
cwlController.SetCWLFeedbackEnabled(true);

// Set starting parameters
cwlController.SetDefaultProfile(myStartingProfile);

// Begin warmup phase (call at trial start)
cwlController.EnableWarmup();
```

### Monitoring During Trial
```csharp
// Current workload level (last inference)
CWLController.CWLLevel level = cwlController.GetCurrentCWLLevel;

// Applied step index (only updates if feedback enabled)
int appliedStep = cwlController.GetCurrentStepIdx;

// Recommended step index (always updated, for logging)
int recommendedStep = cwlController.GetRecommendedStepIdx;

// Current drone parameter values
Dictionary<string, float> limits = cwlController.GetCurrentControlLimits();
// Contains: maxSpeed, maxYawRate, maxPitch, maxRoll, maxAltitudeRate, maxAlpha
```


