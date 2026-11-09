using System;

/// <summary>
/// Which generated layout the scene was built from, for <c>ExperimentRecorder</c>'s session record. Set by the
/// procedural city builder when it builds a layout (Assets/Scripts/Experiment/layouts/&lt;id&gt;.json, written by the
/// Python generator citysim/citygen.py); empty in the tile cities. Static, like <see cref="GoalSession"/>, and cleared
/// on entering play so a session never inherits the previous one's layout.
/// </summary>
public static class CityLayoutInfo
{
    [Serializable]
    public class Record
    {
        public string id = "";
        public string path = "";
        public int seed;
        public string design = "";
        public float size;
        public float p;
        public float s;
        public float f;
        public float alphaAchieved;
        public string scene = "";
        public string generatorVersion = "";
    }

    public static Record Current { get; private set; }

    public static bool HasLayout => Current != null && !string.IsNullOrEmpty(Current.id);

    [UnityEngine.RuntimeInitializeOnLoadMethod(UnityEngine.RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetOnPlay() => Current = null;

    public static void Set(Record record) => Current = record;
}
