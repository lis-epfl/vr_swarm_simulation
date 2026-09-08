using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

/// <summary>
/// Reports and repairs which colliders <see cref="OlfatiSaber"/> will treat as obstacles.
///
/// <para>The avoidance law has exactly one membership rule: <c>Physics.OverlapSphereNonAlloc</c> against
/// the <c>Obstacle</c> layer mask. So an obstacle is any GameObject that both carries a Collider and sits
/// on layer <c>Obstacle</c> — the name, the tag and the prefab it came from are all irrelevant. That
/// makes a stray layer assignment invisible until a drone swerves around something that is not there.</para>
///
/// <para><b>The failure this exists for.</b> ScaledCityWorld (and CityWorld) were authored by setting the
/// layer on the city root with "change children too", so every road plate, footpath, kerb, street light
/// and hydrant landed on <c>Obstacle</c> alongside the buildings. The expensive one is
/// <c>Road_Structure_NNN</c>: a 90.83 x 90.83 x ~0 plate covering a whole patch.
/// <see cref="OlfatiSaber.GetObstacleCylinder"/> takes the circumradius of the axis-aligned bounds, so
/// that flat square becomes a cylinder of radius ~64 units standing over the entire patch — the giant
/// circle in the gizmos, and a repulsion with no gap between patches to fly through. A flat plate is the
/// worst possible input to a bounds-circumradius model, exactly as <see cref="ObstacleCylinderGizmos"/>
/// warns.</para>
///
/// <para>The building families are the ones <see cref="BuildingWidthTuner"/> tunes; keep the two lists in
/// step if the city pack gains another family.</para>
/// </summary>
public static class ObstacleLayerAuditor
{
    private const string k_ObstacleLayerName = "Obstacle";

    // Same four families BuildingWidthTuner scales. Anything else in the pack — roads, ground plates,
    // footpaths, kerbs, street furniture, vegetation — is scenery a drone may fly over or past.
    private static readonly string[] k_BuildingPrefixes =
    {
        "BnP_Small_Building_",
        "BnP_Large_Building_",
        "BnP_Apartment_",
        "Skyscraper_",
    };

    [MenuItem("Tools/Swarm/Audit obstacle layer")]
    public static void Audit()
    {
        int obstacleLayer = LayerMask.NameToLayer(k_ObstacleLayerName);
        if (obstacleLayer < 0)
        {
            Debug.LogError("ObstacleLayerAuditor: no layer named " + k_ObstacleLayerName + ".");
            return;
        }

        List<Collider> onLayer = CollidersOnLayer(obstacleLayer);
        List<Collider> strays = onLayer.Where(c => !IsBuilding(c.gameObject.name)).ToList();
        int buildings = onLayer.Count - strays.Count;

        Debug.Log($"ObstacleLayerAuditor: {onLayer.Count} colliders on '{k_ObstacleLayerName}' — " +
                  $"{buildings} buildings, {strays.Count} not buildings.");

        // Grouped by family rather than listed, because a city is thousands of objects and the reading
        // that matters is "which kinds of thing are in here", ordered by how much space each one steals.
        foreach (var group in strays.GroupBy(c => Family(c.gameObject.name))
                                    .OrderByDescending(g => g.Max(CylinderRadius)))
        {
            Collider worst = group.OrderByDescending(CylinderRadius).First();
            Debug.Log($"  {group.Count(),5} x {group.Key} — largest obstacle cylinder radius " +
                      $"{CylinderRadius(worst):F1} u ({worst.gameObject.name})", worst.gameObject);
        }

        if (strays.Count > 0)
        {
            Debug.Log("Run Tools/Swarm/Restrict obstacle layer to buildings to move these back to Default.");
        }
    }

    [MenuItem("Tools/Swarm/Restrict obstacle layer to buildings")]
    public static void RestrictToBuildings()
    {
        int obstacleLayer = LayerMask.NameToLayer(k_ObstacleLayerName);
        if (obstacleLayer < 0)
        {
            Debug.LogError("ObstacleLayerAuditor: no layer named " + k_ObstacleLayerName + ".");
            return;
        }

        // Only GameObjects that actually carry a Collider are touched. Unity's layer field is per-object
        // and the physics query reads the collider's own object, so demoting a parent would change
        // nothing while quietly rewriting the scene, and promoting a building's parent would not make it
        // an obstacle either.
        int demoted = 0;
        foreach (Collider c in CollidersOnLayer(obstacleLayer))
        {
            if (IsBuilding(c.gameObject.name))
            {
                continue;
            }
            Undo.RecordObject(c.gameObject, "Restrict obstacle layer");
            c.gameObject.layer = 0; // Default
            RecordOverride(c.gameObject);
            demoted++;
        }

        // The other half: a building that was never put on the layer is an invisible hole in the city's
        // avoidance, which reads as "the drones clip that one tower" and is much harder to spot than a
        // spurious obstacle.
        int promoted = 0;
        foreach (Collider c in AllColliders())
        {
            if (c.gameObject.layer == obstacleLayer || !IsBuilding(c.gameObject.name))
            {
                continue;
            }
            Undo.RecordObject(c.gameObject, "Restrict obstacle layer");
            c.gameObject.layer = obstacleLayer;
            RecordOverride(c.gameObject);
            promoted++;
        }

        if (demoted > 0 || promoted > 0)
        {
            EditorSceneManager.MarkAllScenesDirty();
        }

        Debug.Log($"ObstacleLayerAuditor: moved {demoted} non-building colliders off " +
                  $"'{k_ObstacleLayerName}' and {promoted} building colliders onto it. " +
                  "Save the scene to keep this.");
    }

    /// <summary>Every collider in the loaded scenes, inactive ones included.</summary>
    private static Collider[] AllColliders()
    {
        return Object.FindObjectsByType<Collider>(FindObjectsInactive.Include, FindObjectsSortMode.None);
    }

    /// <summary>Every collider currently on <paramref name="layer"/>.</summary>
    private static List<Collider> CollidersOnLayer(int layer)
    {
        return AllColliders().Where(c => c.gameObject.layer == layer).ToList();
    }

    /// <summary>
    /// The radius <see cref="OlfatiSaber"/> would give this collider, for the stock vertical cylinder
    /// axis — the circumradius of the horizontal bounds, which is what makes a flat ground plate so
    /// expensive: it is wide in both horizontal axes and the height it lacks buys nothing back.
    /// </summary>
    private static float CylinderRadius(Collider c)
    {
        Vector3 e = c.bounds.extents;
        return Mathf.Sqrt(e.x * e.x + e.z * e.z);
    }

    private static bool IsBuilding(string objectName)
    {
        foreach (string prefix in k_BuildingPrefixes)
        {
            if (objectName.StartsWith(prefix))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Trailing instance number stripped, so BnP_Street_Light_216 groups with its peers.</summary>
    private static string Family(string objectName)
    {
        int i = objectName.Length;
        while (i > 0 && (char.IsDigit(objectName[i - 1]) || objectName[i - 1] == ' '))
        {
            i--;
        }
        if (i > 0 && objectName[i - 1] == '_')
        {
            i--;
        }
        return i > 0 ? objectName.Substring(0, i) : objectName;
    }

    /// <summary>
    /// Register the layer change as a prefab-instance override. Every city tile is a prefab instance, and
    /// without this a scripted change to one can be dropped when the scene is saved.
    /// </summary>
    private static void RecordOverride(GameObject go)
    {
        if (PrefabUtility.IsPartOfPrefabInstance(go))
        {
            PrefabUtility.RecordPrefabInstancePropertyModifications(go);
        }
    }
}
