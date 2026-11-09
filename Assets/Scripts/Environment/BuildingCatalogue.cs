using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// The ScaledCity building designs, one entry per design: the pack building a procedural city copies,
/// and its collider in the building's own frame. Written by <c>Tools/Swarm/Export building catalogue</c>
/// (<c>BuildingCatalogueExport</c>), read by <c>ProceduralCityBuilder</c>; the same table goes to
/// <c>Assets/Scripts/Experiment/building_catalogue.json</c> for the Python layout generator, which picks
/// the study design from it by name.
///
/// <para>A design is a building name with its instance number removed (<c>BnP_Small_Building_F_017</c> is
/// design <c>BnP_Small_Building_F</c>), the rule <c>ObstacleLayerAuditor</c> groups by. Its source is the
/// first instance across the 48 ScaledCity patches that a copy can be made of: a collider on
/// <c>Obstacle</c>, active with all its parents (the rule <c>DiamondParkBuilder</c> applies to the
/// buildings it copies).</para>
///
/// <para>Pack buildings are authored local +Z up with the pivot at the base, so the collider's local
/// X and Y are the footprint and Z the height; <see cref="Entry.Width"/> and <see cref="Entry.Depth"/>
/// are those two in metres at the ScaledCity scale (the patch hierarchy above a building is unscaled, so
/// its lossy scale is its local scale).</para>
/// </summary>
[CreateAssetMenu(menuName = "Swarm/Building catalogue", fileName = "BuildingCatalogue")]
public class BuildingCatalogue : ScriptableObject
{
    [Serializable]
    public class Entry
    {
        [Tooltip("Building name with the trailing instance number removed.")]
        public string design;
        [Tooltip("BnP_Small_Building_, BnP_Large_Building_, BnP_Apartment_ or Skyscraper_.")]
        public string family;
        [Tooltip("The instance a copy is made of: a GameObject inside a ScaledCity patch prefab.")]
        public GameObject source;
        [Tooltip("The patch prefab the source sits in.")]
        public string patch;
        [Tooltip("BoxCollider size and centre, in the building's local frame (unscaled).")]
        public Vector3 colliderSize;
        public Vector3 colliderCenter;
        [Tooltip("The source's lossy scale in its patch: (width scale, depth scale, height scale).")]
        public Vector3 lossyScale;
        [Tooltip("Instances of this design across the 48 patches (active, on Obstacle).")]
        public int count;

        /// <summary>Footprint along the building's local X, metres.</summary>
        public float Width => Mathf.Abs(colliderSize.x * lossyScale.x);

        /// <summary>Footprint along the building's local Y (world Z at yaw 0), metres.</summary>
        public float Depth => Mathf.Abs(colliderSize.y * lossyScale.y);

        /// <summary>Height along the building's local Z, metres (50 in ScaledCity).</summary>
        public float Height => Mathf.Abs(colliderSize.z * lossyScale.z);

        /// <summary>The local Z scale that makes this design <paramref name="metres"/> tall.</summary>
        public float ZScaleFor(float metres) => Mathf.Abs(colliderSize.z) > 1e-6f ? metres / Mathf.Abs(colliderSize.z) : 0f;
    }

    public List<Entry> entries = new List<Entry>();

    /// <summary>The entry for <paramref name="design"/>, or null.</summary>
    public Entry Find(string design)
    {
        foreach (Entry e in entries)
        {
            if (e.design == design)
            {
                return e;
            }
        }
        return null;
    }
}
