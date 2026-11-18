using System;
using System.IO;
using UnityEditor;
using UnityEditor.Recorder;
using UnityEditor.Recorder.Encoder;
using UnityEditor.Recorder.Input;
using UnityEngine;

/// <summary>
/// Tools/Video: records the footage the instruction videos ask for (Video/README.md).
///
///  * Capture rig at Play: spawns a <see cref="VideoCaptureRig"/> in whatever scene is played, so no
///    scene has to be edited (the setting is per user, in EditorPrefs).
///  * Record at Play: also records the rig's two cameras with the Unity Recorder, from the moment
///    they exist until Play stops, into Video/_recordings/ as
///    &lt;time&gt;_&lt;scene&gt;_overhead.mp4 and &lt;time&gt;_&lt;scene&gt;_interface.mp4 (one take, two angles).
///  * Start / Stop recording now: the same, by hand, during Play (for several takes per session).
///
/// Recording runs at the game's own rate (variable frame rate, capped at nothing), so the flight
/// is not slowed down or sped up by it. Trim the takes and drop them into the video's footage/
/// folder under the names its build/shots.md lists.
/// </summary>
[InitializeOnLoad]
public static class VideoCaptureMenu
{
    const string RigPref = "Video.CaptureRigAtPlay";
    const string RecordPref = "Video.RecordAtPlay";
    const string InterfacePref = "Video.RecordInterface";
    const string OverheadPref = "Video.RecordOverhead";
    const string Root = "Tools/Video/";

    static RecorderController controller;
    static bool waitingToRecord;
    static string currentTake;

    static VideoCaptureMenu()
    {
        EditorApplication.playModeStateChanged += OnPlayMode;
    }

    static bool RigAtPlay
    {
        get => EditorPrefs.GetBool(RigPref, false);
        set => EditorPrefs.SetBool(RigPref, value);
    }

    static bool RecordAtPlay
    {
        get => EditorPrefs.GetBool(RecordPref, false);
        set => EditorPrefs.SetBool(RecordPref, value);
    }

    static bool RecordInterface
    {
        get => EditorPrefs.GetBool(InterfacePref, true);
        set => EditorPrefs.SetBool(InterfacePref, value);
    }

    static bool RecordOverhead
    {
        get => EditorPrefs.GetBool(OverheadPref, true);
        set => EditorPrefs.SetBool(OverheadPref, value);
    }

    public static string RecordingsFolder =>
        Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Video", "_recordings"));

    // -- menu ---------------------------------------------------------------------------------
    [MenuItem(Root + "Capture rig at Play", false, 1)]
    static void ToggleRig()
    {
        RigAtPlay = !RigAtPlay;
        if (RigAtPlay) EnsureTags();
        if (!RigAtPlay) RecordAtPlay = false;
    }

    [MenuItem(Root + "Capture rig at Play", true)]
    static bool ToggleRigValidate()
    {
        Menu.SetChecked(Root + "Capture rig at Play", RigAtPlay);
        return true;
    }

    [MenuItem(Root + "Record at Play", false, 2)]
    static void ToggleRecord()
    {
        RecordAtPlay = !RecordAtPlay;
        if (RecordAtPlay) { RigAtPlay = true; EnsureTags(); }
    }

    [MenuItem(Root + "Record at Play", true)]
    static bool ToggleRecordValidate()
    {
        Menu.SetChecked(Root + "Record at Play", RecordAtPlay);
        return true;
    }

    [MenuItem(Root + "Record the interface camera", false, 20)]
    static void ToggleInterface() => RecordInterface = !RecordInterface;

    [MenuItem(Root + "Record the interface camera", true)]
    static bool ToggleInterfaceValidate()
    {
        Menu.SetChecked(Root + "Record the interface camera", RecordInterface);
        return true;
    }

    [MenuItem(Root + "Record the overhead camera", false, 21)]
    static void ToggleOverhead() => RecordOverhead = !RecordOverhead;

    [MenuItem(Root + "Record the overhead camera", true)]
    static bool ToggleOverheadValidate()
    {
        Menu.SetChecked(Root + "Record the overhead camera", RecordOverhead);
        return true;
    }

    [MenuItem(Root + "Start recording now", false, 40)]
    static void StartNow()
    {
        if (UnityEngine.Object.FindObjectOfType<VideoCaptureRig>() == null) SpawnRig();
        waitingToRecord = true;
    }

    [MenuItem(Root + "Start recording now", true)]
    static bool StartNowValidate() => EditorApplication.isPlaying && (controller == null || !controller.IsRecording());

    [MenuItem(Root + "Stop recording now", false, 41)]
    static void StopNow() => StopRecording();

    [MenuItem(Root + "Stop recording now", true)]
    static bool StopNowValidate() => controller != null && controller.IsRecording();

    [MenuItem(Root + "Add capture tags", false, 60)]
    static void AddTagsMenu() => EnsureTags();

    [MenuItem(Root + "Open recordings folder", false, 61)]
    static void OpenFolder()
    {
        Directory.CreateDirectory(RecordingsFolder);
        EditorUtility.RevealInFinder(RecordingsFolder + Path.DirectorySeparatorChar);
    }

    // -- play mode ----------------------------------------------------------------------------
    static void OnPlayMode(PlayModeStateChange change)
    {
        if (change == PlayModeStateChange.EnteredPlayMode)
        {
            if (RigAtPlay) SpawnRig();
            if (RecordAtPlay) waitingToRecord = true;
            EditorApplication.update -= Tick;
            EditorApplication.update += Tick;
        }
        else if (change == PlayModeStateChange.ExitingPlayMode)
        {
            waitingToRecord = false;
            StopRecording();
            EditorApplication.update -= Tick;
        }
    }

    static void SpawnRig()
    {
        if (UnityEngine.Object.FindObjectOfType<VideoCaptureRig>() != null) return;
        new GameObject("VideoCaptureRig").AddComponent<VideoCaptureRig>();
    }

    static void Tick()
    {
        if (!waitingToRecord || !EditorApplication.isPlaying) return;
        var rig = UnityEngine.Object.FindObjectOfType<VideoCaptureRig>();
        if (rig == null) return;
        bool needOverhead = RecordOverhead && rig.overheadEnabled;
        bool needIface = RecordInterface && rig.interfaceEnabled;
        if ((needOverhead && rig.OverheadCamera == null) || (needIface && rig.InterfaceCamera == null)) return;
        waitingToRecord = false;
        StartRecording(needOverhead, needIface);
    }

    static void StartRecording(bool overhead, bool iface)
    {
        if (!overhead && !iface)
        {
            Debug.LogWarning("[Video] Nothing to record: both cameras are switched off under Tools/Video.");
            return;
        }
        Directory.CreateDirectory(RecordingsFolder);
        string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        currentTake = $"{DateTime.Now:yyyyMMdd_HHmmss}_{scene}";

        var settings = ScriptableObject.CreateInstance<RecorderControllerSettings>();
        if (overhead) settings.AddRecorderSettings(Movie("overhead", VideoCaptureRig.OverheadTag));
        if (iface) settings.AddRecorderSettings(Movie("interface", VideoCaptureRig.InterfaceTag));
        settings.SetRecordModeToManual();
        settings.FrameRatePlayback = FrameRatePlayback.Variable;
        settings.FrameRate = 30f;
        settings.CapFrameRate = false;

        controller = new RecorderController(settings);
        controller.PrepareRecording();
        if (controller.StartRecording())
            Debug.Log($"[Video] Recording take {currentTake} into {RecordingsFolder}");
        else
            Debug.LogError("[Video] The Recorder refused to start; see the console above.");
    }

    static MovieRecorderSettings Movie(string label, string tag)
    {
        var rig = UnityEngine.Object.FindObjectOfType<VideoCaptureRig>();
        var m = ScriptableObject.CreateInstance<MovieRecorderSettings>();
        m.name = label;
        m.Enabled = true;
        m.EncoderSettings = new CoreEncoderSettings
        {
            Codec = CoreEncoderSettings.OutputCodec.MP4,
            EncodingQuality = CoreEncoderSettings.VideoEncodingQuality.High,
        };
        m.ImageInputSettings = new CameraInputSettings
        {
            Source = ImageSource.TaggedCamera,
            CameraTag = tag,
            OutputWidth = rig != null ? rig.width : 1920,
            OutputHeight = rig != null ? rig.height : 1080,
            CaptureUI = false,
        };
        m.AudioInputSettings.PreserveAudio = false;
        m.OutputFile = Path.Combine(RecordingsFolder, $"{currentTake}_{label}");
        return m;
    }

    static void StopRecording()
    {
        if (controller == null) return;
        if (controller.IsRecording())
        {
            controller.StopRecording();
            Debug.Log($"[Video] Saved take {currentTake} in {RecordingsFolder}");
        }
        controller = null;
    }

    // -- tags ---------------------------------------------------------------------------------
    static void EnsureTags()
    {
        var assets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset");
        if (assets == null || assets.Length == 0) return;
        var so = new SerializedObject(assets[0]);
        var tags = so.FindProperty("tags");
        bool changed = false;
        foreach (var tag in new[] { VideoCaptureRig.OverheadTag, VideoCaptureRig.InterfaceTag })
        {
            bool found = false;
            for (int i = 0; i < tags.arraySize; i++)
                if (tags.GetArrayElementAtIndex(i).stringValue == tag) { found = true; break; }
            if (found) continue;
            tags.InsertArrayElementAtIndex(tags.arraySize);
            tags.GetArrayElementAtIndex(tags.arraySize - 1).stringValue = tag;
            changed = true;
        }
        if (changed)
        {
            so.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();
            Debug.Log("[Video] Added the VideoOverhead and VideoInterface tags.");
        }
    }
}
