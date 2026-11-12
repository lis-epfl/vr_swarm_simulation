using System;
using UnityEngine;

/// <summary>
/// The wall materials the scenic city's buildings are drawn in (<see cref="ScenicBuilding"/>, shader
/// <c>Swarm/Scenic Facade</c>): photographed surfaces from Poly Haven (CC0), each a colour map and a map of its normal
/// and roughness, with how many metres one repeat covers. ScenicPatchCityWorldBuilder fills the asset.
///
/// <para>The shader reads them as two texture arrays, one layer per wall, so every building shares one material and the
/// buildings batch. The arrays are not assets: a 14-layer array saved as text is tens of megabytes. They are copied
/// together on the GPU from the imported textures whenever <see cref="Apply"/> finds them missing (at Play, and in the
/// editor when a <see cref="ScenicCity"/> is enabled), which is why every wall's textures must share one size, one
/// format and one mip count; the builder imports them that way.</para>
/// </summary>
[CreateAssetMenu(menuName = "Swarm/Scenic facades", fileName = "ScenicFacades")]
public class ScenicFacades : ScriptableObject
{
    /// <summary>The shader's <c>_ScenicWallMetres</c> has this many entries.</summary>
    public const int MaxWalls = 16;

    [Serializable]
    public class Wall
    {
        public string name = "";
        public Texture2D albedo;
        [Tooltip("R, G: the normal's x and y (OpenGL, as Unity's); B: roughness. Imported linear, not as a normal map.")]
        public Texture2D normal;
        [Tooltip("Metres one repeat of the texture covers on the wall.")]
        public float metres = 2f;
        [Tooltip("A greyscale texture whose colour comes from the building's tint (plaster, painted brick, panels).")]
        public bool paintable;
        [Tooltip("The colour map's mean, linear: a tint of (target / mean) gives a paintable wall the target colour.")]
        public Color mean = Color.grey;
    }

    public Wall[] walls = new Wall[0];
    [Tooltip("Swarm/Scenic Facade: the one material every scenic building is drawn in.")]
    public Material material;
    [Tooltip("Tiling value noise at three scales (R coarse, G, B fine) and vertical streaks (A).")]
    public Texture2D noise;
    [Tooltip("What a window reflects below the roofline across the street: the roofs' height (m), the street's width " +
             "between facades (m), and how bright the reflected facade is.")]
    public Vector3 canyon = new Vector3(50f, 25f, 0.9f);
    [Tooltip("Scales how far the buildings rise above their colliders (roofs at 50 m): 1 as designed, 0 a flat skyline. " +
             "Everything above the collider is seen only from below: the drones' ceiling is under it.")]
    [Range(0f, 1.5f)] public float extraHeight = 1f;

    [NonSerialized] private Texture2DArray albedoArray, normalArray;

    /// <summary>The layer of the wall called <paramref name="wallName"/>, or 0 with a warning.</summary>
    public int Layer(string wallName)
    {
        for (int i = 0; i < walls.Length; i++)
        {
            if (walls[i].name == wallName) return i;
        }
        Debug.LogWarning($"ScenicFacades: no wall called '{wallName}'; using '{(walls.Length > 0 ? walls[0].name : "none")}'.", this);
        return 0;
    }

    /// <summary>The tint that gives wall <paramref name="layer"/> the colour <paramref name="target"/> (sRGB) on
    /// average, no channel above 1.</summary>
    public Color TintFor(int layer, Color target)
    {
        Color mean = walls[layer].mean, t = target.linear;
        return new Color(Mathf.Clamp01(t.r / Mathf.Max(mean.r, 0.01f)), Mathf.Clamp01(t.g / Mathf.Max(mean.g, 0.01f)),
                         Mathf.Clamp01(t.b / Mathf.Max(mean.b, 0.01f)), 1f);
    }

    /// <summary>Makes the texture arrays if they are missing and hands the shader what it reads globally. Cheap to call
    /// again. False (with an error) if the walls cannot be made into arrays.</summary>
    public bool Apply()
    {
        if (walls == null || walls.Length == 0 || walls.Length > MaxWalls)
        {
            Debug.LogError($"ScenicFacades: {(walls == null ? 0 : walls.Length)} walls; it needs 1 to {MaxWalls}.", this);
            return false;
        }
        if (albedoArray == null || normalArray == null || albedoArray.depth != walls.Length)
        {
            Release();
            var albedo = new Texture2D[walls.Length];
            var normal = new Texture2D[walls.Length];
            for (int i = 0; i < walls.Length; i++)
            {
                albedo[i] = walls[i].albedo;
                normal[i] = walls[i].normal;
            }
            albedoArray = Stack(albedo, false, "Scenic wall albedo");
            normalArray = Stack(normal, true, "Scenic wall normal");
            if (albedoArray == null || normalArray == null)
            {
                Release();
                return false;
            }
        }
        var metres = new float[MaxWalls];
        for (int i = 0; i < MaxWalls; i++) metres[i] = i < walls.Length ? Mathf.Max(walls[i].metres, 0.05f) : 1f;
        Shader.SetGlobalTexture("_ScenicWallAlbedo", albedoArray);
        Shader.SetGlobalTexture("_ScenicWallNormal", normalArray);
        Shader.SetGlobalFloatArray("_ScenicWallMetres", metres);
        Shader.SetGlobalTexture("_ScenicNoise", noise != null ? noise : Texture2D.grayTexture);
        Shader.SetGlobalVector("_ScenicCanyon", new Vector4(canyon.x, Mathf.Max(canyon.y, 1f), canyon.z, 0f));
        return true;
    }

    private Texture2DArray Stack(Texture2D[] layers, bool linear, string label)
    {
        Texture2D first = layers[0];
        foreach (Texture2D t in layers)
        {
            if (t == null || first == null || t.width != first.width || t.height != first.height || t.format != first.format ||
                t.mipmapCount != first.mipmapCount)
            {
                Debug.LogError($"ScenicFacades: {label}: every wall needs a texture of one size, format and mip count " +
                               $"({(t == null ? "one is missing" : $"{t.name} is {t.width}x{t.height} {t.format}, {t.mipmapCount} mips")}; " +
                               $"the first is {(first == null ? "missing" : $"{first.width}x{first.height} {first.format}, {first.mipmapCount} mips")}). " +
                               "Tools/Swarm/Build scenic patch city world imports them alike.", this);
                return null;
            }
        }
        var array = new Texture2DArray(first.width, first.height, layers.Length, first.format, first.mipmapCount > 1, linear)
        {
            name = label, wrapMode = TextureWrapMode.Repeat, filterMode = FilterMode.Trilinear, anisoLevel = 8,
            hideFlags = HideFlags.HideAndDontSave,
        };
        for (int i = 0; i < layers.Length; i++)
        {
            for (int mip = 0; mip < first.mipmapCount; mip++) Graphics.CopyTexture(layers[i], 0, mip, array, i, mip);
        }
        return array;
    }

    private void Release()
    {
        if (albedoArray != null) DestroyNow(albedoArray);
        if (normalArray != null) DestroyNow(normalArray);
        albedoArray = null;
        normalArray = null;
    }

    private static void DestroyNow(UnityEngine.Object o)
    {
        if (Application.isPlaying) Destroy(o);
        else DestroyImmediate(o);
    }

    private void OnDisable() => Release();
}
