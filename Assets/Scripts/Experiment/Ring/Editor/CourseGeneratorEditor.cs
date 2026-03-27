using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(CourseGenerator))]
public class CourseGeneratorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        // Draw the default inspector properties
        DrawDefaultInspector();

        EditorGUILayout.Space(10);

        CourseGenerator generator = (CourseGenerator)target;

        if (GUILayout.Button("Generate Course (Preview Only)", GUILayout.Height(40)))
        {
            generator.GenerateCoursePlan();
        }
    }
}
