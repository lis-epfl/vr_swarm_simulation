using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// Widens the streets that run between the city's patches, in any scene that uses the Modular City Pack
/// tiles — ScaledCityWorld, CityWorld, CrowdWorld, or a fork of them. Drop it on <c>gameManager</c> and
/// move <see cref="blockScale"/> down.
///
/// <para><b>What makes this possible.</b> <c>Road_Structure_NNN</c> is a full-tile asphalt plate,
/// 90.83 x 90.83, and the footpath is a separate raised slab sitting <i>on top of it</i> — so there is
/// already road underneath the pavement. Shrinking a tile's block content about its own centre, while
/// leaving the plate and the grid pitch alone, therefore exposes more asphalt at the tile rim rather
/// than a hole in the ground. Two tiles meet at each boundary, so the street between them widens by
/// twice what each block gives up, and the plate's baked centre line stays in the middle of it. No new
/// geometry, no filler, no seams.</para>
///
/// <para><b>The arithmetic.</b> The footpath/kerb content spans 76.2 of the 90.8304 tile (the pack is
/// modelled in inches: 3000 in of 3576), leaving a 7.315 asphalt margin at each edge and so a
/// <b>14.63-unit street today</b>. At a block scale <c>s</c> the street is
/// <c>90.8304 - 76.2 * s</c> — 18.4 at 0.95, 22.3 at 0.9, 26.1 at 0.85, 29.9 at 0.8.
/// <see cref="ReportStreetWidth"/> prints both that figure and one measured off the live geometry.</para>
///
/// <para><b>Buildings keep their size.</b> Only their positions are scaled, so a tower stays a tower and
/// the swarm's obstacle field is unchanged in extent. The cost is that buildings crowd together by
/// <c>(1 - s)</c> of their spacing: fine at 0.9 given ScaledCityWorld already narrows them to 25%, but
/// below about 0.85 expect the denser blocks to interpenetrate. That, not the street width, is what
/// puts a practical floor on the knob — judge it by eye.</para>
///
/// <para>Props are moved and not scaled for the same reason, which is what keeps a street light at the
/// kerb: the lamp at |x| = 37.81 and the footpath edge at 38.1 both move inward by the same factor, so
/// the 0.29 between them is preserved at every <c>s</c>.</para>
///
/// <para><b>Tune it in edit mode, before pressing Play.</b> The city pack marks every tile object
/// <c>Batching Static</c>, so entering Play combines their meshes and from then on the renderer ignores
/// the transform — a runtime change would move colliders without moving anything you can see. As in
/// <see cref="BuildingWidthTuner"/>, <see cref="ApplyScale"/> refuses to run once the city is batched and
/// says so. In edit mode there is no batch, the Scene view updates live, and the value carries into Play
/// without saving. Saving makes it permanent and writes a <c>localPosition</c> override per moved object,
/// so use <see cref="RestoreAuthoredLayout"/> rather than dragging back by eye.</para>
///
/// <para>Goal patches instantiated at runtime by <see cref="GoalPatchReplacer"/> are brought up to the
/// city's block scale on the first frame (<see cref="MatchNewPatches"/>), exactly as
/// <see cref="BuildingWidthTuner.MatchNewBuildings"/> does for their buildings.</para>
///
/// <para><b>Known consequences, accepted deliberately.</b> The interior <c>Roads_Street_NN</c> strips
/// narrow with the block (6.71 carriageway to 6.04 at s = 0.9) and so does the 1.524 footpath — both are
/// baked into the same per-tile mesh as everything else, so neither can be held fixed while the block
/// shrinks around it. The plate's baked road-edge marking stays at 38.1 while the kerb moves in, leaving
/// a faint double edge line.</para>
///
/// <para><b>Interaction with <see cref="BuildingWidthTuner"/>:</b> that component writes building
/// <c>localScale</c> and this one writes building <c>localPosition</c>, so the two compose without
/// fighting and either may be used alone.</para>
/// </summary>
[ExecuteAlways]
[DisallowMultipleComponent]
public class StreetWidthTuner : MonoBehaviour
{
    /// <summary>Tile pitch, and the size of the <c>Road_Structure</c> plate. 3576 in at 0.0254 m/in.</summary>
    private const float k_TilePitch = 90.8304f;

    /// <summary>How much of that tile the authored footpath/kerb content spans. 3000 in.</summary>
    private const float k_AuthoredBlockSpan = 76.2f;

    [Tooltip("Scales each tile's block content about its own centre; 1 = unchanged. Lower means wider " +
             "streets. The resulting street width is 90.8304 - 76.2 x this, i.e. 14.6 at 1, 22.3 at " +
             "0.9, 29.9 at 0.8. Below ~0.85 buildings start to interpenetrate.")]
    [Range(0.6f, 1.2f)]
    [SerializeField] private float blockScale = 1f;

    [Tooltip("Optional, and normally left empty — an empty root scans every loaded scene, which is what " +
             "picks up goal patches and any other city geometry spawned at runtime. Set it (e.g. to " +
             "City_Pack_01_Scaled) only to confine the change to one city in a scene that has more than one.")]
    [SerializeField] private Transform cityRoot;

    [Tooltip("The baked per-tile shell. These are the only objects whose mesh is scaled as well as " +
             "moved — everything else in a tile keeps its size and is only repositioned.")]
    [SerializeField] private string[] shellNamePrefixes =
    {
        "FootPath_",
        "Footpath_",
        "Carbs_",
        "Roads_Street_",
    };

    [Tooltip("The full-tile asphalt plate, left untouched — it is what keeps the ground continuous and " +
             "supplies the road surface the widened street is drawn on.")]
    [SerializeField] private string[] ignoredNamePrefixes =
    {
        "Road_Structure_",
    };

    [Tooltip("Also move the trees, green belts and garden furniture that hang off the city root rather " +
             "than off a tile, each about the tile it is nearest. Off leaves them overhanging the road.")]
    [SerializeField] private bool moveLooseScenery = true;

    [Tooltip("Log a line each time the scale is re-applied. Off by default: dragging the slider applies " +
             "once per frame and would spam the console.")]
    [SerializeField] private bool logChanges = false;

    // What the city is currently scaled by. Serialised alongside blockScale so the two survive a scene
    // save together: a rescan divides this back out to recover the authored layout, and the knob keeps
    // the same meaning across sessions instead of treating a saved tuning as the new baseline. Hidden
    // because blockScale already shows it — they are equal except mid-apply.
    [HideInInspector]
    [SerializeField] private float appliedBlockScale = 1f;

    /// <summary>
    /// One object the tuning moves, and the frame it is measured in.
    ///
    /// <para>Tile content is mapped in its own tile's local space, which is the node the FBX Z-up import
    /// rotated: there the ground plane is local X/Y and height is local Z. Loose scenery has no tile to
    /// belong to and is mapped in world space instead, about the nearest tile centre. Both are the same
    /// scale-about-a-point; only the frame differs, and <see cref="frame"/> being null is what says which.</para>
    /// </summary>
    private struct Item
    {
        public Transform transform;
        public Transform frame;        // the tile's content node; null = mapped in world space
        public Vector3 authoredPos;    // position in that frame, with any applied scale divided back out
        public Vector3 centre;         // the point it is scaled about, in the same frame
        public int frameHeightAxis;    // component of a position in that frame that is height; never scaled
        public bool isShell;           // shell meshes are resized as well as moved
        public Vector3 authoredScale;  // shell only
        public int ownHeightAxis;      // shell only: which of its own local axes points at the sky
    }

    private readonly List<Item> items = new List<Item>();
    private bool scanned;
    private int batchedCount;      // objects whose mesh has been combined into a static batch
    private bool warnedAboutBatch; // the batching message is printed once, not once per frame

    // The tiles that existed before any Start ran. Non-null only in a play session that has tuning to
    // propagate; cleared after the first Update, after which the field costs one reference compare
    // per frame.
    private HashSet<Transform> preexistingTiles;

    /// <summary>
    /// Snapshot the tiles the scene loaded with, so <see cref="MatchNewPatches"/> can tell them from ones
    /// spawned later. Awake runs before every Start, which is what makes the snapshot exclude the goal
    /// patches <see cref="GoalPatchReplacer"/> instantiates.
    /// </summary>
    private void Awake()
    {
        if (!Application.isPlaying || appliedBlockScale == 1f)
        {
            return; // untuned scene: a spawned tile is already the right size
        }

        preexistingTiles = new HashSet<Transform>();
        foreach (Transform kerb in CollectKerbs())
        {
            preexistingTiles.Add(kerb);
        }
    }

    private void Update()
    {
        if (preexistingTiles != null)
        {
            // First frame of a tuned play session: every Start has run, so anything that was going to
            // spawn has spawned.
            MatchNewPatches();
            preexistingTiles = null;
        }

        if (blockScale == appliedBlockScale)
        {
            return; // the common case: one float compare per frame, nothing else
        }

        if (!gameObject.scene.IsValid())
        {
            return; // opened as a prefab asset rather than in a scene; nothing to tune
        }

        if (!scanned)
        {
            // Deferred to Update rather than Start so GoalPatchReplacer, which instantiates goal patches
            // in its own Start, has already placed them and their content is scanned too.
            Rescan();
        }

        ApplyScale();
    }

#if UNITY_EDITOR
    /// <summary>
    /// Apply an inspector edit straight away instead of waiting for an edit-mode <see cref="Update"/>
    /// tick, which only happens when the editor decides to repaint. The work is deferred to
    /// <c>delayCall</c> because scanning the scene and writing transforms inside a validation callback is
    /// not safe — Unity may still be deserialising.
    /// </summary>
    private void OnValidate()
    {
        if (Application.isPlaying || blockScale == appliedBlockScale)
        {
            return;
        }
        EditorApplication.delayCall += ApplyAfterValidate;
    }

    /// <summary>Deferred half of <see cref="OnValidate"/>; idempotent, so repeats during a drag are free.</summary>
    private void ApplyAfterValidate()
    {
        if (this == null || Application.isPlaying || blockScale == appliedBlockScale)
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
        ApplyScale();
    }
#endif

    /// <summary>
    /// The block scale. Assigning applies it immediately, so a keyboard handler or experiment script can
    /// drive it without waiting a frame.
    /// </summary>
    public float BlockScale
    {
        get => blockScale;
        set
        {
            blockScale = Mathf.Max(0.05f, value);
            if (blockScale == appliedBlockScale)
            {
                return;
            }
            if (!scanned)
            {
                Rescan();
            }
            ApplyScale();
        }
    }

    /// <summary>The street width this scale produces, in world units.</summary>
    public static float StreetWidthFor(float scale)
    {
        return k_TilePitch - k_AuthoredBlockSpan * scale;
    }

    /// <summary>Put the city back to the layout it was authored with.</summary>
    [ContextMenu("Restore authored layout")]
    public void RestoreAuthoredLayout()
    {
        BlockScale = 1f;
    }

    /// <summary>
    /// Print the street width, both as the arithmetic predicts it and as the live geometry measures it.
    /// The measured figure is the tile pitch less the widest footpath slab's horizontal extent, which is
    /// the same quantity a tape measure across a boundary would give, and it is worth having because it
    /// reads the scene rather than this component's own bookkeeping.
    /// </summary>
    [ContextMenu("Report street width")]
    public void ReportStreetWidth()
    {
        float predicted = StreetWidthFor(blockScale);

        float widestSlab = 0f;
        string slabName = null;
        foreach (Transform kerb in CollectKerbs())
        {
            Transform tile = kerb.parent;
            if (tile == null)
            {
                continue;
            }
            foreach (Renderer r in tile.GetComponentsInChildren<Renderer>(true))
            {
                if (!HasPrefix(r.gameObject.name, "FootPath_") && !HasPrefix(r.gameObject.name, "Footpath_"))
                {
                    continue;
                }
                Vector3 size = r.bounds.size;
                float span = Mathf.Max(size.x, size.z);
                if (span > widestSlab)
                {
                    widestSlab = span;
                    slabName = r.gameObject.name;
                }
            }
        }

        string measured = slabName == null
            ? "no FootPath slab found to measure against"
            : $"measured {k_TilePitch - widestSlab:F2} u (tile pitch {k_TilePitch:F2} less {slabName} " +
              $"at {widestSlab:F2})";

        Debug.Log($"StreetWidthTuner: block scale {blockScale:F3} — street {predicted:F2} u; {measured}.", this);
    }

    /// <summary>
    /// Re-collect the tiles and re-capture their authored positions. <see cref="Update"/> does this
    /// automatically the first time it is needed; call it after spawning or destroying city geometry.
    /// Safe at any time — the applied scale is divided back out of the live positions, so repeated scans
    /// never compound.
    /// </summary>
    [ContextMenu("Rescan city")]
    public void Rescan()
    {
        items.Clear();
        batchedCount = 0;

        float undo = appliedBlockScale > 0f ? 1f / appliedBlockScale : 1f;

        List<Transform> kerbs = CollectKerbs();
        List<Transform> tiles = new List<Transform>();
        List<Vector3> tileCentresWorld = new List<Vector3>();
        HashSet<Transform> tileContent = new HashSet<Transform>();

        int shells = 0;
        int moved = 0;

        foreach (Transform kerb in kerbs)
        {
            Transform tile = kerb.parent;
            if (tile == null)
            {
                continue; // a kerb with no tile node is not something this knows how to map
            }

            // The kerb's own pivot is the block's centre: the FootPath, Carbs and Roads_Street meshes are
            // all authored about it, and the footpath slab spans +/-38.1 around it. Deliberately not the
            // tile node's origin, which is not the same point in every tile — MC_Patch_32 has its whole
            // block (kerb, footpath, buildings and asphalt plate alike) baked 318 units off its own
            // pivot, an authoring quirk inherited from the unscaled City_Pack_01 prefab. Reading the
            // centre off the kerb follows the content wherever it was baked.
            //
            // The height component comes along and is then discarded everywhere below: seven tiles carry
            // a baked vertical offset of -27 to -50 in this frame, cancelled by a matching offset on the
            // tile instance, and scaling that would drop them through the ground.
            Vector3 centre = kerb.localPosition;

            // Which component of that is height has to be measured, not hardcoded. 48 of the 49 tiles
            // hang their content off the -90-about-X node the FBX Z-up import produced, where height is
            // local Z — but MC_Patch_32 leaves that node at identity and pushes the rotation down onto
            // each child instead, so its block is measured in a Y-up frame. Assuming Z there would scale
            // the tile's height and preserve one of its horizontal axes, i.e. exactly backwards.
            int frameHeightAxis = FindHeightAxis(tile, out _);

            tiles.Add(tile);
            tileCentresWorld.Add(tile.TransformPoint(centre));

            foreach (Renderer renderer in tile.GetComponentsInChildren<Renderer>(true))
            {
                Transform t = renderer.transform;
                if (IsIgnored(t.name))
                {
                    continue; // the asphalt plate: the one thing that must not move
                }

                tileContent.Add(t);
                if (renderer.isPartOfStaticBatch)
                {
                    batchedCount++;
                }

                Vector3 live = tile.InverseTransformPoint(t.position);
                Vector3 authored = centre + (live - centre) * undo;
                authored[frameHeightAxis] = live[frameHeightAxis]; // height is never touched

                bool isShell = IsShell(t.name);
                Vector3 authoredScale = t.localScale;
                int ownHeightAxis = 0;

                if (isShell)
                {
                    shells++;
                    ownHeightAxis = FindHeightAxis(t, out _);
                    if (ownHeightAxis != 0) { authoredScale.x *= undo; }
                    if (ownHeightAxis != 1) { authoredScale.y *= undo; }
                    if (ownHeightAxis != 2) { authoredScale.z *= undo; }
                }
                else
                {
                    moved++;
                }

                items.Add(new Item
                {
                    transform = t,
                    frame = tile,
                    authoredPos = authored,
                    centre = centre,
                    frameHeightAxis = frameHeightAxis,
                    isShell = isShell,
                    authoredScale = authoredScale,
                    ownHeightAxis = ownHeightAxis,
                });
            }
        }

        int scenery = 0;
        Transform sceneryRoot = cityRoot != null ? cityRoot : CommonAncestor(tiles);
        if (moveLooseScenery && tiles.Count > 0 && sceneryRoot != null)
        {
            // Trees, green belts and garden furniture hang off the city root rather than off a tile, and
            // a good number of them stand at the kerb. Mapped per leaf rather than per group because a
            // Green_Belt_Tile group can straddle more than one tile.
            foreach (Renderer renderer in LooseSceneryRenderers(sceneryRoot, tileContent))
            {
                Transform t = renderer.transform;
                Vector3 centre = NearestTileCentre(t.position, tileCentresWorld);

                Vector3 live = t.position;
                Vector3 authored = centre + (live - centre) * undo;
                authored.y = live.y; // world frame here, so height is world Y

                if (renderer.isPartOfStaticBatch)
                {
                    batchedCount++;
                }

                items.Add(new Item
                {
                    transform = t,
                    frame = null,
                    authoredPos = authored,
                    centre = centre,
                    frameHeightAxis = 1,
                    isShell = false,
                    authoredScale = t.localScale,
                    ownHeightAxis = 0,
                });
                scenery++;
            }
        }
        else if (moveLooseScenery && tiles.Count > 0)
        {
            Debug.LogWarning("StreetWidthTuner: the city tiles share no common parent, so the loose trees " +
                             "and green belts cannot be attributed to a city and are left where they are. " +
                             "Set cityRoot to fix this.", this);
        }

        scanned = true;

        string where = cityRoot != null ? cityRoot.name : "the scene";
        if (items.Count == 0)
        {
            Debug.LogWarning($"StreetWidthTuner: found no city tiles in {where} — a tile is recognised by " +
                             "its Carbs_NN kerb object. Check shellNamePrefixes and cityRoot. Street " +
                             "widening will do nothing.", this);
        }
        else
        {
            Debug.Log($"StreetWidthTuner: tracking {tiles.Count} tiles in {where} — {shells} shell meshes, " +
                      $"{moved} buildings and props, {scenery} loose scenery" +
                      (batchedCount > 0 ? $", {batchedCount} of them in a static batch" : "") + ".", this);
        }
    }

    /// <summary>Write the current <see cref="blockScale"/> to every tracked object.</summary>
    private void ApplyScale()
    {
        if (batchedCount > 0)
        {
            // Their meshes were combined when Play started, so moving them now would drag the colliders
            // away from the visible city — invisible walls in the middle of the road, which is worse than
            // not widening at all. Refuse the whole change rather than half-applying it.
            if (!warnedAboutBatch)
            {
                warnedAboutBatch = true;
                Debug.LogError(
                    $"StreetWidthTuner: {batchedCount} of {items.Count} city objects are part of a static " +
                    $"batch, so their meshes were combined when Play started and no runtime move can " +
                    $"shift them. Block scale left at {appliedBlockScale:F3}x. Set blockScale before " +
                    $"entering Play — it widens the streets live in the Scene view and carries into Play " +
                    $"— or clear 'Batching Static' on the tiles to tune during Play at the cost of their " +
                    $"batching.", this);
            }
            return;
        }

        float scale = Mathf.Max(0.05f, blockScale);
        int missing = 0;

        for (int i = 0; i < items.Count; i++)
        {
            Item item = items[i];
            if (item.transform == null)
            {
                missing++; // a tile GoalPatchReplacer destroyed; harmless, and Rescan clears it out
                continue;
            }

            Vector3 target = item.centre + (item.authoredPos - item.centre) * scale;
            target[item.frameHeightAxis] = item.authoredPos[item.frameHeightAxis];

            if (item.frame == null)
            {
                item.transform.position = target;
            }
            else if (item.transform.parent == item.frame)
            {
                item.transform.localPosition = target; // the common case, and exact
            }
            else
            {
                // A prop under Props_Patch_NN. The container itself carries no renderer, so it is never
                // moved and this only has to place the leaf.
                item.transform.position = item.frame.TransformPoint(target);
            }

            if (item.isShell)
            {
                Vector3 s = item.authoredScale;
                if (item.ownHeightAxis != 0) { s.x *= scale; }
                if (item.ownHeightAxis != 1) { s.y *= scale; }
                if (item.ownHeightAxis != 2) { s.z *= scale; }
                item.transform.localScale = s;
            }

            RecordEditModeChange(item.transform);
        }

        appliedBlockScale = scale;
        blockScale = scale;
        MarkSceneDirtyInEditMode();

        if (logChanges)
        {
            Debug.Log($"StreetWidthTuner: block scale {scale:F3}x authored on {items.Count - missing} " +
                      $"objects; streets now {StreetWidthFor(scale):F2} u.", this);
        }
    }

    /// <summary>
    /// Bring tiles spawned since <see cref="Awake"/> up to the block scale the rest of the city is at.
    /// They come straight out of a prefab asset — goal patches, via <see cref="GoalPatchReplacer"/> — so
    /// they carry the authored layout and would otherwise stand in a widened city with the old narrow
    /// streets, which is exactly the mismatch an edit-mode-only tuning produces.
    ///
    /// This is not a scale *change*, so it is not subject to the static-batch refusal in
    /// <see cref="ApplyScale"/>: a runtime instance is never part of a batch, and moving it brings the
    /// scene into agreement rather than splitting its visuals from its colliders. Its live layout is the
    /// authored one by definition — nothing has tuned it before — so the scale is applied directly.
    /// </summary>
    private void MatchNewPatches()
    {
        float scale = appliedBlockScale;
        int matched = 0;

        foreach (Transform kerb in CollectKerbs())
        {
            if (preexistingTiles.Contains(kerb))
            {
                continue; // loaded with the scene, so it already carries the applied scale
            }

            Transform tile = kerb.parent;
            if (tile == null)
            {
                continue;
            }

            Vector3 centre = kerb.localPosition;
            int frameHeightAxis = FindHeightAxis(tile, out _);

            foreach (Renderer renderer in tile.GetComponentsInChildren<Renderer>(true))
            {
                Transform t = renderer.transform;
                if (IsIgnored(t.name) || renderer.isPartOfStaticBatch)
                {
                    continue;
                }

                Vector3 live = tile.InverseTransformPoint(t.position);
                Vector3 target = centre + (live - centre) * scale;
                target[frameHeightAxis] = live[frameHeightAxis];

                if (t.parent == tile)
                {
                    t.localPosition = target;
                }
                else
                {
                    t.position = tile.TransformPoint(target);
                }

                if (IsShell(t.name))
                {
                    int ownHeightAxis = FindHeightAxis(t, out _);
                    Vector3 s = t.localScale;
                    if (ownHeightAxis != 0) { s.x *= scale; }
                    if (ownHeightAxis != 1) { s.y *= scale; }
                    if (ownHeightAxis != 2) { s.z *= scale; }
                    t.localScale = s;
                }
            }
            matched++;
        }

        if (matched > 0)
        {
            Debug.Log($"StreetWidthTuner: matched {matched} newly spawned tiles to the city's " +
                      $"{scale:F3}x block scale.", this);
        }
    }

    /// <summary>
    /// Every tile's kerb object, which is how a tile is recognised at all. Every patch in the pack has a
    /// <c>Carbs_NN</c> child — including the seven that have no <c>FootPath</c> — and its parent is the
    /// tile's content node, the one the FBX Z-up import rotated.
    /// </summary>
    private List<Transform> CollectKerbs()
    {
        Transform[] candidates = cityRoot != null
            ? cityRoot.GetComponentsInChildren<Transform>(true)
            : FindObjectsByType<Transform>(FindObjectsInactive.Include, FindObjectsSortMode.None);

        List<Transform> found = new List<Transform>();
        foreach (Transform t in candidates)
        {
            if (HasPrefix(t.name, "Carbs_"))
            {
                found.Add(t);
            }
        }
        return found;
    }

    /// <summary>
    /// City geometry that belongs to no tile — the trees, green belts and garden furniture parented
    /// straight to the city root. Anything already claimed by a tile is excluded, so nothing is mapped
    /// twice.
    ///
    /// <para>This is scoped to <paramref name="sceneryRoot"/> and never to the whole scene, which is the
    /// difference between this and <see cref="CollectKerbs"/>. A kerb is identified by name, so scanning
    /// everything is safe; loose scenery is identified only by *not* being something else, so an
    /// unscoped scan would sweep in the drones, the arena and the curved screen and scale their positions
    /// toward the nearest city tile.</para>
    /// </summary>
    private List<Renderer> LooseSceneryRenderers(Transform sceneryRoot, HashSet<Transform> tileContent)
    {
        List<Renderer> found = new List<Renderer>();
        foreach (Renderer r in sceneryRoot.GetComponentsInChildren<Renderer>(true))
        {
            if (tileContent.Contains(r.transform) || IsIgnored(r.gameObject.name))
            {
                continue;
            }
            if (r.transform.IsChildOf(transform))
            {
                continue; // whatever this component is sitting on is not city scenery
            }
            found.Add(r);
        }
        return found;
    }

    /// <summary>
    /// The nearest transform every tile hangs under — the city root, without needing it to be named or
    /// assigned. Null if the tiles do not share one, in which case loose scenery is left alone rather
    /// than guessed at.
    /// </summary>
    private static Transform CommonAncestor(List<Transform> nodes)
    {
        if (nodes.Count == 0)
        {
            return null;
        }

        Transform common = nodes[0].parent;
        for (int i = 1; i < nodes.Count && common != null; i++)
        {
            while (common != null && !nodes[i].IsChildOf(common))
            {
                common = common.parent;
            }
        }
        return common;
    }

    /// <summary>The tile centre nearest <paramref name="worldPos"/>, measured horizontally.</summary>
    private static Vector3 NearestTileCentre(Vector3 worldPos, List<Vector3> centres)
    {
        Vector3 best = centres[0];
        float bestSqr = float.MaxValue;
        for (int i = 0; i < centres.Count; i++)
        {
            float dx = centres[i].x - worldPos.x;
            float dz = centres[i].z - worldPos.z;
            float sqr = dx * dx + dz * dz;
            if (sqr < bestSqr)
            {
                bestSqr = sqr;
                best = centres[i];
            }
        }
        return best;
    }

    private bool IsShell(string objectName)
    {
        return HasAnyPrefix(objectName, shellNamePrefixes);
    }

    private bool IsIgnored(string objectName)
    {
        return HasAnyPrefix(objectName, ignoredNamePrefixes);
    }

    private static bool HasAnyPrefix(string objectName, string[] prefixes)
    {
        if (prefixes == null)
        {
            return false;
        }

        foreach (string prefix in prefixes)
        {
            if (HasPrefix(objectName, prefix))
            {
                return true;
            }
        }
        return false;
    }

    private static bool HasPrefix(string objectName, string prefix)
    {
        return !string.IsNullOrEmpty(prefix) && objectName.StartsWith(prefix);
    }

    /// <summary>
    /// Which local axis of <paramref name="t"/> points at the sky (0 = X, 1 = Y, 2 = Z), and how well
    /// (|cos| to world up, 1 = exactly vertical). The pack's tiles hang under a node rotated -90 about X,
    /// so their content is local +Z up — but measuring beats assuming, and it keeps this working on any
    /// other geometry. Same rule as <see cref="BuildingWidthTuner"/> uses.
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
    /// Register an edit-mode change as a prefab-instance override. Without this a scripted change to a
    /// prefab instance — which every city tile is — can be dropped when the scene is saved.
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
    /// localPosition override per moved object.
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
