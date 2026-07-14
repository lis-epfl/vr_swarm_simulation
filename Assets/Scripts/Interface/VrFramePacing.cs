using UnityEngine;

// Frame-pacing hygiene for the VR scenes. The XR compositor paces frames for
// the headset; Unity's own main-display vsync only stalls the render thread
// (vSyncCount 2 on the "Ultra" quality level capped the sim at half refresh),
// so force it off and leave the frame rate uncapped. Runs once at startup,
// after any quality-level selection, without needing a scene object.
public static class VrFramePacing
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Init()
    {
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
    }
}
