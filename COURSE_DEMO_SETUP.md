# Course Demo Animation - Scene Setup Instructions

The code for the demo is complete. Follow these steps in the Unity Editor to wire everything up:

## Step 1: Create the DemoDot GameObject

1. In the Scene hierarchy, find or create an empty GameObject named `DemoDot`
   - If it doesn't exist: Right-click in hierarchy → Create Empty → rename to `DemoDot`
   - Suggested location: as a sibling of `CoursePathVisual` or under the experiment manager

2. Add the `CourseDemo` component:
   - Select `DemoDot` → Inspector → Add Component → `CourseDemo`

3. Wire the references:
   - **pathVisual**: Drag the GameObject that has `CoursePathVisual` component into this field
   - **dotMaterial**: Assign `Assets/Materials/dots.mat` (green, emissive material)

4. Tuning (optional):
   - **demoSpeed**: Default 10 m/s (inspector-tunable at runtime)
   - **dotRadius**: Default 0.35 (controls size of the demo sphere)

## Step 2: Create the DemoCamera GameObject

1. Create an empty GameObject named `DemoCamera`
   - Right-click in hierarchy → Create Empty → rename to `DemoCamera`

2. Add the `Camera` component:
   - Select `DemoCamera` → Inspector → Add Component → Camera
   - Set **Target Display** to **5** (same display as FlyingTaskOverlay)
   - Set **Depth** to **-2** (so it renders behind the UI Canvas)
   - Set **Near** to 0.3, **Far** to 1000 (or your preferred values)

3. Add the `DemoFollowCamera` script:
   - Inspector → Add Component → `DemoFollowCamera`

4. Wire the references:
   - **target**: Drag the `DemoDot` GameObject into this field
   - **backDistance**: 8 (default, distance behind the dot)
   - **height**: 3 (default, height above the dot)
   - **smoothSpeed**: 5 (default, camera follow smoothness)

## Step 3: Update FlyingTaskOverlay Canvas

1. In the hierarchy, find the `FlyingTaskOverlay` Canvas GameObject

2. Update the Canvas component:
   - **Render Mode**: ScreenSpaceCamera (should already be set)
   - **Target Display**: 5 (should already be set)
   - **Camera**: Drag the `DemoCamera` GameObject here (this makes the Canvas render on top of the 3D demo)

## Step 4: Add FlyingInstructions to State Configs

The demo uses the existing `stateConfigs` system for automatic lifecycle management (no manual FSM calls needed):

1. Select the GameObject with `ExperimentFSMRacingGate` component (usually `ExperimentManager` or similar)

2. In the Inspector, find **Step Visuals** section → `stateConfigs` list

3. Add a new entry:
   - Click the `+` button
   - **State**: Select `FlyingInstructions`
   - **enableOnState**: Add both `FlyingTaskOverlay` and `DemoCamera` GameObjects

When `FlyingInstructions` state is entered, the `stateConfigs` system will automatically:
- Call `SetActive(true)` on both GameObjects
- Trigger `OnEnable()` in `CourseDemo`, which calls `StartDemo()`
- Trigger `OnEnable()` in `DemoFollowCamera`, which initializes the follow camera

When exiting the state, `OnDisable()` calls `StopDemo()` automatically.

## Verification Checklist

- [ ] `DemoDot` has `CourseDemo` script with references wired
- [ ] `DemoCamera` has `Camera` component set to Display 5, Depth -2
- [ ] `DemoCamera` has `DemoFollowCamera` script with target wired to `DemoDot`
- [ ] `FlyingTaskOverlay` Canvas has Camera field set to `DemoCamera`
- [ ] `FlyingInstructions` added to stateConfigs with both `FlyingTaskOverlay` and `DemoCamera` GameObjects
- [ ] Code compiles with no errors

## Testing in Play Mode

1. Enter Play mode
2. Transition to `FlyingInstructions` state (click "Next" in operator UI or call FSM manually)
3. You should see:
   - On Display 5: A cyan path line with a green dot moving along it, with a camera following behind
   - The "Flying task" text overlay appears on top
4. Adjust `demoSpeed` in the Inspector to change the dot's speed (in real-time)
5. Transition to `FlyingPractice` to stop the demo

## Troubleshooting

**Demo doesn't start when entering FlyingInstructions:**
- Verify `FlyingInstructions` is added to `stateConfigs` with both GameObjects
- Check that both GameObjects are initially inactive (or they won't trigger OnEnable)
- Look at Console for "[CourseDemo] Demo started" log message

**Demo dot doesn't appear:**
- Verify `CourseDemo.dotMaterial` is assigned to `dots.mat`
- Check Console for "[CourseDemo] Created demo dot" log
- Ensure Display 5 is active on your hardware setup

**Camera not following:**
- Verify `DemoFollowCamera.target` is set to `DemoDot`
- Check that `DemoCamera` has a `Camera` component (not just the script)
- Check Console for any errors in `DemoFollowCamera.LateUpdate()`

**Path not visible:**
- Check that `CoursePathVisual` component exists and has `gateManager` assigned
- Verify a course has been generated (should happen in `FlyingInstructions.EnterState`)
- Check the `CourseDemo.pathVisual` reference is wired

**Canvas not showing on top:**
- Ensure `FlyingTaskOverlay` Canvas has **Camera** field set to `DemoCamera`
- Check that `DemoCamera` Depth is lower (-2) than any other cameras rendering to Display 5
- Verify `FlyingTaskOverlay` RenderMode is `ScreenSpaceCamera`
