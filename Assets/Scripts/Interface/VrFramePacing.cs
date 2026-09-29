using UnityEngine;

// Frame-pacing hygiene for the VR scenes. The XR compositor paces frames for
// the headset; Unity's own main-display vsync only stalls the render thread
// (vSyncCount 2 on the "Ultra" quality level capped the sim at half refresh),
// so force it off and leave the frame rate uncapped. Runs once at startup,
// after any quality-level selection, without needing a scene object.
//
// It also caps how much simulated time one frame may catch up on. The project's Maximum
// Allowed Timestep is 0.333 s, so a single hitch (a GC, a first-frame spawn) cashes out as up
// to 16 back-to-back physics steps in the next frame, which is itself long enough to miss the
// headset's 13.9 ms budget and so gets the app locked at half rate. 0.1 s is still five steps
// of the 0.02 s tick -- nothing short of a real stall is affected -- and a longer stall now
// slows game time for a moment instead of bursting.
public static class VrFramePacing
{
    private const float MaxCatchUpSeconds = 0.1f;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Init()
    {
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
        Time.maximumDeltaTime = MaxCatchUpSeconds;
    }
}
