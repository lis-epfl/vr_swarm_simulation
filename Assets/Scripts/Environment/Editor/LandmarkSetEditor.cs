using UnityEditor;
using UnityEngine;

/// <summary><see cref="LandmarkSet"/>'s inspector: the fields, how to edit the set, and the <i>Place models</i> and
/// <i>Check</i> buttons (<see cref="LandmarkBuilder"/>).</summary>
[CustomEditor(typeof(LandmarkSet))]
public class LandmarkSetEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        EditorGUILayout.Space();
        EditorGUILayout.HelpBox(
            "Ground: select a tile under Ground and sculpt or paint it with the Terrain tools.\n" +
            "Lake: select Lake (N) and drag its spline's knots; the water follows. To start it over, delete it and run " +
            "Tools > Swarm > Landmarks > Add to open scene.\n" +
            "Mast and turbines: set a model slot above and press Place models (clear it to get the generated one back).\n" +
            "Keep edits in Landmarks.prefab (edit it there, or Overrides > Apply All): a scene rebuild re-instantiates " +
            "the prefab. Sculpting is stored in the tiles' own TerrainData, so it is kept either way.",
            MessageType.Info);
        using (new EditorGUI.DisabledScope(EditorApplication.isPlayingOrWillChangePlaymode))
        {
            if (GUILayout.Button("Place models")) LandmarkBuilder.PlaceModelsAndSave((LandmarkSet)target);
            if (GUILayout.Button(new GUIContent("Check", "Whether each landmark still shows in the feeds from across the " +
                                                         "city, and that nothing in the set collides or casts shadows.")))
            {
                LandmarkBuilder.CheckAndLog((LandmarkSet)target);
            }
        }
    }
}
