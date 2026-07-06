using System;
using UnityEngine;

/// <summary>
/// Reads Meta Quest Pro eye-tracking data through OVRPlugin and exposes raw gaze
/// vectors (per-eye and combined) in world space, gated by a confidence threshold.
///
/// This is the foundation for eye tracking as a performance metric: it only surfaces
/// the data via a read-only static API (mirroring PyUniSharingFast.BodyYawDegrees).
/// File logging and gaze-target raycasting are layered on top of this later.
///
/// Design notes:
/// - Polls OVRPlugin.GetEyeGazesState directly rather than using OVREyeGaze, because a
///   single call returns both eyes with validity + confidence, which OVREyeGaze hides.
/// - Raw eye poses are relative to the OVRCameraRig tracking space, so they are
///   converted to world space through the cached trackingSpace transform.
/// - Fails gracefully (warns + disables) on headsets without eye tracking, so the sim
///   still runs on non-Quest-Pro devices and in the editor.
/// </summary>
public class EyeGazeTracker : MonoBehaviour
{
    /// <summary>A single world-space gaze ray with its reliability.</summary>
    public struct GazeSample
    {
        public Vector3 Origin;      // world-space eye position
        public Vector3 Direction;   // world-space, normalized look direction
        public float Confidence;    // 0..1 reliability reported by the runtime
        public bool IsValid;        // false when tracking is lost / below threshold
    }

    [Header("Eye Tracking")]
    [SerializeField]
    [Tooltip("Reference frame the raw eye poses are relative to (OVRCameraRig.trackingSpace). " +
             "Auto-found from the OVRCameraRig in the scene if left empty.")]
    private Transform trackingSpace;

    [SerializeField]
    [Range(0f, 1f)]
    [Tooltip("Gaze samples below this confidence are reported as invalid.")]
    private float confidenceThreshold = 0.5f;

    /// <summary>Left-eye gaze in world space (last polled frame).</summary>
    public static GazeSample Left { get; private set; }

    /// <summary>Right-eye gaze in world space (last polled frame).</summary>
    public static GazeSample Right { get; private set; }

    /// <summary>
    /// Combined gaze: midpoint origin and averaged direction of the valid eyes.
    /// This is the single ray most consumers (debug cursor, later metrics) should read.
    /// </summary>
    public static GazeSample Combined { get; private set; }

    /// <summary>True once eye tracking has successfully started on this device.</summary>
    public static bool EyeTrackingActive { get; private set; }

    private const OVRPermissionsRequester.Permission EyeTrackingPermission =
        OVRPermissionsRequester.Permission.EyeTracking;

    private OVRPlugin.EyeGazesState eyeGazesState;
    private Action<string> onPermissionGranted;
    private bool started;

    private void Awake()
    {
        onPermissionGranted = OnPermissionGranted;
    }

    private void OnEnable()
    {
        // If the permission isn't granted yet, TryStartEyeTracking requests it and we
        // start once the granted callback fires; nothing else to do here.
        TryStartEyeTracking();
    }

    private void OnDisable()
    {
        OVRPermissionsRequester.PermissionGranted -= onPermissionGranted;
        if (started)
        {
            OVRPlugin.StopEyeTracking();
            started = false;
        }
        EyeTrackingActive = false;
    }

    private bool TryStartEyeTracking()
    {
        if (started) return true;

        if (!OVRPermissionsRequester.IsPermissionGranted(EyeTrackingPermission))
        {
            // Ask for the runtime permission and resume from the granted callback.
            // (No-op / auto-granted off-Android, e.g. over Quest Link, where the ray still works.)
            OVRPermissionsRequester.PermissionGranted -= onPermissionGranted;
            OVRPermissionsRequester.PermissionGranted += onPermissionGranted;
            OVRPermissionsRequester.Request(new[] { EyeTrackingPermission });
            return false;
        }

        if (!OVRPlugin.StartEyeTracking())
        {
            Debug.LogWarning("[EyeGazeTracker] Failed to start eye tracking. " +
                             "Gaze data will be unavailable (headset may not support eye tracking).");
            enabled = false;
            return false;
        }

        started = true;
        EyeTrackingActive = true;
        return true;
    }

    private void OnPermissionGranted(string permissionId)
    {
        if (permissionId == OVRPermissionsRequester.GetPermissionId(EyeTrackingPermission))
        {
            OVRPermissionsRequester.PermissionGranted -= onPermissionGranted;
            TryStartEyeTracking();
        }
    }

    private void Update()
    {
        if (!started) return;
        if (!ResolveTrackingSpace()) return;

        if (!OVRPlugin.GetEyeGazesState(OVRPlugin.Step.Render, -1, ref eyeGazesState))
            return;

        Left = ReadEye(OVREyeGaze.EyeId.Left);
        Right = ReadEye(OVREyeGaze.EyeId.Right);
        Combined = CombineEyes(Left, Right);
    }

    // Convert one eye's tracking-space pose into a world-space gaze sample.
    private GazeSample ReadEye(OVREyeGaze.EyeId eye)
    {
        OVRPlugin.EyeGazeState state = eyeGazesState.EyeGazes[(int)eye];

        if (!state.IsValid || state.Confidence < confidenceThreshold)
        {
            return new GazeSample { IsValid = false };
        }

        // Pose is relative to the tracking space; lift it into world space through the rig.
        OVRPose local = state.Pose.ToOVRPose();
        Vector3 forward = local.orientation * Vector3.forward;

        return new GazeSample
        {
            Origin = trackingSpace.TransformPoint(local.position),
            Direction = (trackingSpace.rotation * forward).normalized,
            Confidence = state.Confidence,
            IsValid = true
        };
    }

    // Merge the two eyes: midpoint origin, averaged direction. Falls back to whichever
    // single eye is valid, and is invalid only when neither eye is tracked.
    private static GazeSample CombineEyes(GazeSample l, GazeSample r)
    {
        if (l.IsValid && r.IsValid)
        {
            return new GazeSample
            {
                Origin = (l.Origin + r.Origin) * 0.5f,
                Direction = (l.Direction + r.Direction).normalized,
                Confidence = (l.Confidence + r.Confidence) * 0.5f,
                IsValid = true
            };
        }

        if (l.IsValid) return l;
        if (r.IsValid) return r;
        return new GazeSample { IsValid = false };
    }

    // Lazily locate the OVRCameraRig tracking space, mirroring PyUniSharingFast.FindHeadTransform.
    private bool ResolveTrackingSpace()
    {
        if (trackingSpace != null) return true;

        OVRCameraRig rig = FindObjectOfType<OVRCameraRig>();
        if (rig != null && rig.trackingSpace != null)
        {
            trackingSpace = rig.trackingSpace;
            return true;
        }
        return false;
    }
}
