using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// Tunes the horizontal footprint of a city's buildings, in any scene that uses the Modular City Pack
/// tiles — CityWorld, ScaledCityWorld, CrowdWorld, or a fork of them. Drop it on <c>gameManager</c> and
/// move <see cref="widthScale"/>.
///
/// <see cref="widthScale"/> multiplies whatever width the scene was authored with, so 1 always means
/// "leave it as it is" and the component is inert until touched. It is deliberately *not* a fraction of
/// the original pack width: ScaledCityWorld already bakes 0.25 into its prefabs and CityWorld bakes 1,
/// so an absolute scale would need a different per-scene baseline and would silently quadruple the
/// buildings in the wrong scene. In ScaledCityWorld, 4 restores the original pack width and 0.5 halves
/// the current 25%. Height is never touched, only the two horizontal axes.
///
/// <para><b>Tune it in edit mode, before pressing Play — that is the path that works everywhere.</b>
/// The city pack marks every building <c>Batching Static</c> (in the original prefabs as well as the
/// ScaledCity forks), so entering Play combines their meshes and from then on the renderer ignores the
/// transform: a runtime scale change would move each building's collider without moving the building
/// you can see. Rather than hand you invisible walls, <see cref="ApplyWidth"/> refuses to run once the
/// buildings are batched and says so. In edit mode there is no batch, so the slider resizes the city
/// live in the Scene view, and the value carries into Play without saving the scene. Saving is only
/// needed to make it permanent — and does write a <c>localScale</c> override per building into the
/// scene, so use <see cref="RestoreAuthoredWidth"/> rather than dragging back by eye if you change
/// your mind.</para>
///
/// <para>Live tuning *during* Play is possible, at a cost: clear <c>Batching Static</c> on the building
/// objects (they are the four families below) and they stop being combined, so this component drives
/// them every frame — a few hundred extra draw calls in exchange. Anything instantiated at runtime is
/// never batched and is always tunable during Play, which is why goal patches were the one thing that
/// responded before this was understood.</para>
///
/// <para>Geometry spawned at runtime is brought up to the city's width on the first frame
/// (<see cref="MatchNewBuildings"/>). Goal patches come out of their prefab at the authored width, so
/// without this they stand in a tuned city at the wrong size — the one thing that visibly disagreed
/// while tuning was edit-mode only.</para>
///
/// <para><b>Cost.</b> Nothing runs unless the value changes: <see cref="Update"/> is one float compare
/// (two while the runtime-spawn check is still pending). The scene is not scanned until the first
/// change, or twice at the start of a play session that has a tuning to propagate. A change writes one
/// <c>localScale</c> per building. Obstacle avoidance needs no rescan — <see cref="OlfatiSaber"/> reads
/// <c>Collider.bounds</c> every frame and a BoxCollider scales with its transform.</para>
///
/// <para><b>Caveat.</b> A building shrinks toward its own pivot, and ~22 of the pack's buildings have a
/// footprint that is not centred on that pivot, so those shift laterally by a few units rather than
/// staying put.</para>
/// </summary>
[ExecuteAlways]
[DisallowMultipleComponent]
public class BuildingWidthTuner : MonoBehaviour
{
    [Tooltip("Multiplies the building width this scene was authored with; 1 = unchanged. Height is " +
             "unaffected. In ScaledCityWorld (already narrowed to 25% of the pack) 4 is the original " +
             "width; in CityWorld 0.25 reproduces ScaledCityWorld's narrowing.")]
    [Range(0.05f, 8f)]
    [SerializeField] private float widthScale = 1f;

    [Tooltip("Optional, and normally left empty — an empty root scans every loaded scene, which is what " +
             "picks up goal patches and any other city geometry spawned at runtime. Set it (e.g. to " +
             "City_Pack_01_Scaled) only to confine tuning to one city in a scene that has more than one.")]
    [SerializeField] private Transform cityRoot;

    [Tooltip("Name prefixes identifying buildings. Anything else in the tiles — roads, footpaths, " +
             "ground plates, street lights and other props — is left untouched.")]
    [SerializeField] private string[] buildingNamePrefixes =
    {
        "BnP_Small_Building_",
        "BnP_Large_Building_",
        "BnP_Apartment_",
        "Skyscraper_",
    };

    [Tooltip("Log a line each time the width is re-applied. Off by default: dragging the slider applies " +
             "once per frame and would spam the console.")]
    [SerializeField] private bool logChanges = false;

    // What the buildings in this scene are currently scaled by. Serialised alongside widthScale so the
    // two survive a scene save together: a rescan then divides this back out to recover the authored
    // width, and the knob keeps the same meaning across sessions instead of treating a saved tuning as
    // the new baseline. Hidden because widthScale already shows it — they are equal except mid-apply.
    [HideInInspector]
    [SerializeField] private float appliedWidthScale = 1f;

    /// <summary>A building, the width it was authored at, and the axis that points at the sky.</summary>
    private struct Building
    {
        public Transform transform;
        public Vector3 authoredScale; // local scale with any applied tuning divided back out
        public int heightAxis;        // 0 = local X, 1 = local Y, 2 = local Z; never scaled
    }

    private readonly List<Building> buildings = new List<Building>();
    private bool scanned;
    private int batchedCount;      // buildings whose mesh has been combined into a static batch
    private bool warnedAboutBatch; // the batching message is printed once, not once per frame

    // The buildings that existed before any Start ran, i.e. everything already carrying the applied
    // width. Non-null only in a play session that has tuning to propagate; cleared after the first
    // Update, after which the field costs one reference compare per frame.
    private HashSet<Transform> preexisting;

    /// <summary>
    /// Snapshot the buildings the scene loaded with, so <see cref="MatchNewBuildings"/> can tell them
    /// from ones spawned later. Awake runs before every Start, which is what makes the snapshot exclude
    /// the goal patches <see cref="GoalPatchReplacer"/> instantiates.
    /// </summary>
    private void Awake()
    {
        if (!Application.isPlaying || appliedWidthScale == 1f)
        {
            return; // untuned scene: a spawned building is already the right width
        }

        List<Transform> found = CollectBuildings();
        preexisting = new HashSet<Transform>(found.Count);
        foreach (Transform t in found)
        {
            preexisting.Add(t);
        }
    }

    private void Update()
    {
        if (preexisting != null)
        {
            // First frame of a tuned play session: every Start has run, so anything that was going to
            // spawn has spawned.
            MatchNewBuildings();
            preexisting = null;
        }

        if (widthScale == appliedWidthScale)
        {
            return; // the common case: one float compare per frame, nothing else
        }

        if (!gameObject.scene.IsValid())
        {
            return; // opened as a prefab asset rather than in a scene; nothing to tune
        }

        if (!scanned)
        {
            // Deferred to Update rather than Start so GoalPatchReplacer, which instantiates goal
            // patches in its own Start, has already placed them and their buildings are scanned too.
            Rescan();
        }

        ApplyWidth();
    }

#if UNITY_EDITOR
    /// <summary>
    /// Apply an inspector edit straight away instead of waiting for an edit-mode <see cref="Update"/>
    /// tick, which only happens when the editor decides to repaint. The work is deferred to
    /// <c>delayCall</c> because scanning the scene and writing transforms inside a validation callback
    /// is not safe — Unity may still be deserialising.
    /// </summary>
    private void OnValidate()
    {
        if (Application.isPlaying || widthScale == appliedWidthScale)
        {
            return;
        }
        EditorApplication.delayCall += ApplyAfterValidate;
    }

    /// <summary>Deferred half of <see cref="OnValidate"/>; idempotent, so repeats during a drag are free.</summary>
    private void ApplyAfterValidate()
    {
        if (this == null || Application.isPlaying || widthScale == appliedWidthScale)
        {
            return; // component deleted, or another callback in the same drag already applied it
        }
        if (!gameObject.scene.IsValid())
        {
            return;
        }
        if (!scanned)
        {
            Rescan();
        }
        ApplyWidth();
    }
#endif

    /// <summary>
    /// The width multiplier. Assigning applies it immediately, so a keyboard handler or experiment
    /// script can drive it without waiting a frame.
    /// </summary>
    public float WidthScale
    {
        get => widthScale;
        set
        {
            widthScale = Mathf.Max(0.001f, value);
            if (widthScale == appliedWidthScale)
            {
                return;
            }
            if (!scanned)
            {
                Rescan();
            }
            ApplyWidth();
        }
    }

    /// <summary>Put the buildings back to the width the scene was authored with.</summary>
    [ContextMenu("Restore authored width")]
    public void RestoreAuthoredWidth()
    {
        WidthScale = 1f;
    }

    /// <summary>
    /// Re-collect the buildings and re-capture their authored scales. <see cref="Update"/> does this
    /// automatically the first time it is needed; call it after spawning or destroying city geometry.
    /// Safe at any time — the applied multiplier is divided back out of the live scale, so repeated
    /// scans never compound.
    /// </summary>
    [ContextMenu("Rescan buildings")]
    public void Rescan()
    {
        buildings.Clear();
        batchedCount = 0;

        float undo = appliedWidthScale > 0f ? 1f / appliedWidthScale : 1f;

        int offAxis = 0;
        foreach (Transform t in CollectBuildings())
        {
            int heightAxis = FindHeightAxis(t, out float verticality);
            if (verticality < 0.9f)
            {
                offAxis++; // no local axis points at the sky; the least-wrong one is used anyway
            }

            Vector3 authored = t.localScale;
            if (heightAxis != 0) { authored.x *= undo; }
            if (heightAxis != 1) { authored.y *= undo; }
            if (heightAxis != 2) { authored.z *= undo; }

            // False in edit mode, and false for anything Instantiate'd at runtime. True once Unity has
            // combined a Batching Static mesh, after which the renderer ignores this transform.
            if (t.TryGetComponent(out Renderer renderer) && renderer.isPartOfStaticBatch)
            {
                batchedCount++;
            }

            buildings.Add(new Building { transform = t, authoredScale = authored, heightAxis = heightAxis });
        }

        scanned = true;

        string where = cityRoot != null ? cityRoot.name : "the scene";
        if (buildings.Count == 0)
        {
            Debug.LogWarning($"BuildingWidthTuner: found no buildings in {where}; check buildingNamePrefixes " +
                             "and cityRoot. Width tuning will do nothing.", this);
        }
        else
        {
            Debug.Log($"BuildingWidthTuner: tracking {buildings.Count} buildings in {where}" +
                      (batchedCount > 0 ? $", {batchedCount} of them in a static batch" : "") +
                      (offAxis > 0 ? $" ({offAxis} with no clearly vertical local axis)" : "") + ".", this);
        }
    }

    /// <summary>Write the current <see cref="widthScale"/> to every tracked building's horizontal axes.</summary>
    private void ApplyWidth()
    {
        if (batchedCount > 0)
        {
            // Their meshes were combined when Play started, so scaling them now would move the colliders
            // and leave the visible buildings where they are — solid walls the pilot cannot see, which is
            // worse than not tuning at all. Refuse the whole change rather than half-applying it.
            if (!warnedAboutBatch)
            {
                warnedAboutBatch = true;
                Debug.LogError(
                    $"BuildingWidthTuner: {batchedCount} of {buildings.Count} buildings are part of a static batch, " +
                    $"so their meshes were combined when Play started and no runtime scale change can move them. " +
                    $"Width left at {appliedWidthScale:F3}x. Set widthScale before entering Play — it resizes the " +
                    $"city live in the Scene view and carries into Play — or clear 'Batching Static' on the " +
                    $"buildings to tune during Play at the cost of their batching.", this);
            }
            return;
        }

        float scale = Mathf.Max(0.001f, widthScale);
        int missing = 0;

        for (int i = 0; i < buildings.Count; i++)
        {
            Building b = buildings[i];
            if (b.transform == null)
            {
                missing++; // a tile GoalPatchReplacer destroyed; harmless, and Rescan clears it out
                continue;
            }

            Vector3 s = b.authoredScale;
            if (b.heightAxis != 0) { s.x *= scale; }
            if (b.heightAxis != 1) { s.y *= scale; }
            if (b.heightAxis != 2) { s.z *= scale; }
            b.transform.localScale = s;

            RecordEditModeChange(b.transform);
        }

        appliedWidthScale = scale;
        widthScale = scale;
        MarkSceneDirtyInEditMode();

        if (logChanges)
        {
            Debug.Log($"BuildingWidthTuner: width {scale:F3}x authored on {buildings.Count - missing} buildings.", this);
        }
    }

    /// <summary>
    /// Bring buildings spawned since <see cref="Awake"/> up to the width the rest of the city is at.
    /// They come straight out of a prefab asset — goal patches, via <see cref="GoalPatchReplacer"/> —
    /// so they carry the authored width and would otherwise stand next to a tuned city at the wrong
    /// size, which is what the edit-mode-only tuning looked like from the cockpit.
    ///
    /// This is not a width *change*, so it is not subject to the static-batch refusal in
    /// <see cref="ApplyWidth"/>: a runtime instance is never part of a batch, and scaling it brings the
    /// scene into agreement rather than splitting its visuals from its colliders. Their live scale is
    /// the authored one by definition — nothing has tuned them before — so it is multiplied directly.
    /// </summary>
    private void MatchNewBuildings()
    {
        float scale = appliedWidthScale;
        int matched = 0;

        foreach (Transform t in CollectBuildings())
        {
            if (preexisting.Contains(t))
            {
                continue; // loaded with the scene, so it already carries the applied width
            }
            if (t.TryGetComponent(out Renderer renderer) && renderer.isPartOfStaticBatch)
            {
                continue; // cannot be moved visually; leave its collider where its mesh is
            }

            int heightAxis = FindHeightAxis(t, out _);
            Vector3 s = t.localScale;
            if (heightAxis != 0) { s.x *= scale; }
            if (heightAxis != 1) { s.y *= scale; }
            if (heightAxis != 2) { s.z *= scale; }
            t.localScale = s;
            matched++;
        }

        if (matched > 0)
        {
            Debug.Log($"BuildingWidthTuner: matched {matched} newly spawned buildings to the city's " +
                      $"{scale:F3}x width.", this);
        }
    }

    /// <summary>Every building under <see cref="cityRoot"/>, or in every loaded scene if it is unset.</summary>
    private List<Transform> CollectBuildings()
    {
        Transform[] candidates = cityRoot != null
            ? cityRoot.GetComponentsInChildren<Transform>(true)
            : FindObjectsByType<Transform>(FindObjectsInactive.Include, FindObjectsSortMode.None);

        List<Transform> found = new List<Transform>();
        foreach (Transform t in candidates)
        {
            if (IsBuilding(t.name))
            {
                found.Add(t);
            }
        }
        return found;
    }

    /// <summary>True if <paramref name="objectName"/> starts with one of the building family prefixes.</summary>
    private bool IsBuilding(string objectName)
    {
        if (buildingNamePrefixes == null)
        {
            return false;
        }

        foreach (string prefix in buildingNamePrefixes)
        {
            if (!string.IsNullOrEmpty(prefix) && objectName.StartsWith(prefix))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Which local axis of <paramref name="t"/> points at the sky (0 = X, 1 = Y, 2 = Z), and how well
    /// (|cos| to world up, 1 = exactly vertical). The pack's buildings are all local +Z up — an FBX Z-up
    /// import artefact — but measuring beats assuming, and it keeps this working on any other geometry.
    /// </summary>
    private static int FindHeightAxis(Transform t, out float verticality)
    {
        float x = Mathf.Abs(Vector3.Dot(t.right, Vector3.up));
        float y = Mathf.Abs(Vector3.Dot(t.up, Vector3.up));
        float z = Mathf.Abs(Vector3.Dot(t.forward, Vector3.up));

        if (x >= y && x >= z) { verticality = x; return 0; }
        if (y >= z) { verticality = y; return 1; }
        verticality = z;
        return 2;
    }

    /// <summary>
    /// Register an edit-mode scale change as a prefab-instance override. Without this a scripted change
    /// to a prefab instance — which every city tile is — can be dropped when the scene is saved.
    /// </summary>
    private static void RecordEditModeChange(Transform t)
    {
#if UNITY_EDITOR
        if (!Application.isPlaying && PrefabUtility.IsPartOfPrefabInstance(t))
        {
            PrefabUtility.RecordPrefabInstancePropertyModifications(t);
        }
#endif
    }

    /// <summary>
    /// Flag the scene as modified after an edit-mode change, so the tuning is not silently lost when the
    /// scene is closed. It carries into Play without saving; saving makes it permanent and writes a
    /// localScale override per building.
    /// </summary>
    private void MarkSceneDirtyInEditMode()
    {
#if UNITY_EDITOR
        if (!Application.isPlaying && gameObject.scene.IsValid())
        {
            EditorSceneManager.MarkSceneDirty(gameObject.scene);
        }
#endif
    }
}
