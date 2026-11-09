using System.Collections.Generic;
using System.Linq;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// Staggers the city's rows of tiles, so the streets crossing them stop lining up into long open corridors and a
/// flight has to weave between buildings. Drop it on <c>gameManager</c> and move <see cref="Stagger"/>.
///
/// <para><b>What moves.</b> Rows are found from the tiles' kerbs, never their transforms (see
/// <see cref="CityTiles"/>), and numbered from 0 at the city's south edge (west edge, for north–south rows). The
/// chosen rows slide along their own length by <see cref="Stagger"/> tiles; with <see cref="Centred"/> they take
/// half of it and the other rows take the other half the opposite way, which keeps the outline centred. Only tile
/// roots are moved, so everything a tile owns — its block, and its tied Scenery, Street and Verge — goes with it.
/// That is why this <b>refuses to run while the city root still holds loose scenery</b>: the pack's street trees
/// and plates would be left standing where the tiles used to be. Run Tools/Swarm/Tie city scenery to tiles first.</para>
///
/// <para><b>Edit mode only</b>, for the same reason as the tuners: the city is static-batched in Play, and moving a
/// batched tile moves its colliders but not what you see. Like <see cref="StreetWidthTuner"/> it applies live as
/// the value changes and records what it applied, so the stagger keeps its meaning across saves;
/// <see cref="RestoreAuthoredGrid"/> puts the tiles back exactly.</para>
///
/// <para><b>Consequences, by design.</b> A half-tile stagger turns every crossroads on a shifted row into two
/// T-junctions, and each shifted row sticks out half a tile at one end of the city and leaves a half-tile notch at
/// the other (the notch shows whatever lies under the city; <see cref="Centred"/> spreads it over both ends). A
/// median strip on the line between two rows belongs to one of them (the southern, or western), so once the two
/// are staggered it runs across the mouths of the other row's cross streets, where they now end in T-junctions.
/// <see cref="GoalPatchReplacer"/> needs nothing — it reads the kerbs, and
/// its adjacency test measures the staggered neighbours correctly — but a run recorded on the unstaggered grid no
/// longer replays: half its goals stand on a shifted row.</para>
/// </summary>
[ExecuteAlways]
[DisallowMultipleComponent]
public class CityRowOffsetter : MonoBehaviour
{
    public enum RowDirection
    {
        EastWest,   // rows of tiles sharing a Z position, sliding along X
        NorthSouth, // rows of tiles sharing an X position, sliding along Z
    }

    [Tooltip("How far alternate rows slide along their length, in tiles, relative to the rows beside them. 0 is the " +
             "authored grid; 0.5 is a brick pattern, each tile centred on the street between two tiles of the next row.")]
    [Range(-1f, 1f)]
    [SerializeField] private float stagger = 0f;

    [Tooltip("Which way the rows run. EastWest rows are tiles sharing a Z position, and slide along X.")]
    [SerializeField] private RowDirection rows = RowDirection.EastWest;

    [Tooltip("Slide the odd rows (counting from 0 at the south, or west, edge) rather than the even ones.")]
    [SerializeField] private bool shiftOddRows = true;

    [Tooltip("Split the stagger: the chosen rows slide by half of it and the others by half the opposite way, so " +
             "the city's outline stays centred instead of growing on one side.")]
    [SerializeField] private bool centred = false;

    [Tooltip("Optional: the city root. Empty uses the city in this component's scene.")]
    [SerializeField] private Transform cityRoot;

    [Tooltip("Log a line each time the stagger is re-applied. Off by default: dragging the slider applies once per " +
             "frame and would spam the console.")]
    [SerializeField] private bool logChanges = false;

    // What the tiles currently stand at, serialised with the settings so a rescan-free re-apply always starts from
    // the truth: the next apply first undoes exactly this, then applies the new settings, as one move per tile.
    [HideInInspector] [SerializeField] private float appliedStagger = 0f;
    [HideInInspector] [SerializeField] private RowDirection appliedRows = RowDirection.EastWest;
    [HideInInspector] [SerializeField] private bool appliedShiftOddRows = true;
    [HideInInspector] [SerializeField] private bool appliedCentred = false;

    private bool warnedRefusal; // a refusal is printed once, not once per frame

    public float Stagger
    {
        get => stagger;
        set { stagger = Mathf.Clamp(value, -1f, 1f); Apply(); }
    }

    public RowDirection Rows
    {
        get => rows;
        set { rows = value; Apply(); }
    }

    public bool ShiftOddRows
    {
        get => shiftOddRows;
        set { shiftOddRows = value; Apply(); }
    }

    public bool Centred
    {
        get => centred;
        set { centred = value; Apply(); }
    }

    private bool IsApplied =>
        stagger == appliedStagger && rows == appliedRows && shiftOddRows == appliedShiftOddRows && centred == appliedCentred;

    private void Update()
    {
        if (!IsApplied && gameObject.scene.IsValid())
        {
            Apply();
        }
    }

#if UNITY_EDITOR
    /// <summary>Apply an inspector edit straight away; deferred for the same reason as StreetWidthTuner's.</summary>
    private void OnValidate()
    {
        if (!Application.isPlaying && !IsApplied)
        {
            EditorApplication.delayCall += ApplyAfterValidate;
        }
    }

    private void ApplyAfterValidate()
    {
        if (this != null && !Application.isPlaying && !IsApplied && gameObject.scene.IsValid())
        {
            Apply();
        }
    }
#endif

    /// <summary>Put every tile back on the authored grid.</summary>
    [ContextMenu("Restore authored grid")]
    public void RestoreAuthoredGrid()
    {
        Stagger = 0f;
    }

    /// <summary>
    /// Move the tiles to the current settings. Returns false, and moves nothing, if it cannot: during Play, with no
    /// city, or while the city root holds scenery no tile owns.
    /// </summary>
    public bool Apply()
    {
        if (IsApplied)
        {
            return true;
        }
        if (Application.isPlaying)
        {
            Refuse("the city is static-batched in Play, so moving a tile would move its colliders but not what you " +
                   "see. Set the stagger in edit mode; it carries into Play.");
            return false;
        }

        string error;
        CityTiles.City city = cityRoot != null
            ? CityTiles.FindCity(cityRoot, out error)
            : CityTiles.FindCity(gameObject.scene, out error);
        if (city == null)
        {
            Refuse(error);
            return false;
        }

        // Undo what is applied, then apply what is asked, as one move per tile. Rows are numbered on the positions
        // the undo leaves, since sliding along one direction leaves the rows of that direction where they were but
        // scrambles those of the other.
        Vector3 east = Flat(city.Root.right);
        Vector3 north = Flat(city.Root.forward);
        Vector3[] restored = new Vector3[city.Tiles.Count];
        Vector3[] moves = new Vector3[city.Tiles.Count];

        int[] appliedRowOf = RowIndices(city.Centres.ToArray(), appliedRows, east, north);
        for (int i = 0; i < city.Tiles.Count; i++)
        {
            moves[i] = -Shift(appliedRowOf[i], appliedStagger, appliedRows, appliedShiftOddRows, appliedCentred, east, north);
            restored[i] = city.Centres[i] + moves[i];
        }

        int[] rowOf = RowIndices(restored, rows, east, north);
        bool anyMove = false;
        for (int i = 0; i < city.Tiles.Count; i++)
        {
            moves[i] += Shift(rowOf[i], stagger, rows, shiftOddRows, centred, east, north);
            anyMove |= moves[i].sqrMagnitude > 1e-8f;
        }

        if (anyMove)
        {
            List<Transform> loose = city.LooseChildren();
            if (loose.Count > 0)
            {
                Refuse($"{loose.Count} objects under {city.Root.name} belong to no tile " +
                       $"({string.Join(", ", loose.Take(3).Select(t => t.name))}{(loose.Count > 3 ? ", ..." : "")}) and " +
                       "would be left behind. Run Tools/Swarm/Tie city scenery to tiles first.");
                return false;
            }

            for (int i = 0; i < city.Tiles.Count; i++)
            {
                if (moves[i].sqrMagnitude > 1e-8f)
                {
                    city.Tiles[i].position += moves[i];
                    RecordEditModeChange(city.Tiles[i]);
                }
            }
        }

        appliedStagger = stagger;
        appliedRows = rows;
        appliedShiftOddRows = shiftOddRows;
        appliedCentred = centred;
        warnedRefusal = false;
        MarkSceneDirtyInEditMode();

        if (logChanges)
        {
            Debug.Log($"CityRowOffsetter: {(shiftOddRows ? "odd" : "even")} {rows} rows staggered by {stagger:F3} tiles" +
                      (centred ? ", centred" : "") + $" across {city.Tiles.Count} tiles.", this);
        }
        return true;
    }

    /// <summary>
    /// Each position's row: its distance across the rows from the first row, in whole pitches — so a missing row
    /// leaves a gap in the numbering rather than renumbering every row after it.
    /// </summary>
    private static int[] RowIndices(Vector3[] positions, RowDirection direction, Vector3 east, Vector3 north)
    {
        Vector3 across = direction == RowDirection.EastWest ? north : east;
        float first = positions.Min(p => Vector3.Dot(p, across));
        return positions.Select(p => Mathf.RoundToInt((Vector3.Dot(p, across) - first) / CityTiles.Pitch)).ToArray();
    }

    private static Vector3 Shift(int row, float stagger, RowDirection direction, bool oddRows, bool centred,
                                 Vector3 east, Vector3 north)
    {
        bool chosen = (row % 2 == 1) == oddRows;
        float tiles = centred ? (chosen ? 0.5f : -0.5f) * stagger : (chosen ? stagger : 0f);
        return (direction == RowDirection.EastWest ? east : north) * (tiles * CityTiles.Pitch);
    }

    private void Refuse(string reason)
    {
        if (!warnedRefusal)
        {
            warnedRefusal = true;
            Debug.LogError($"CityRowOffsetter: {reason} Stagger left at {appliedStagger:F3}.", this);
        }
    }

    private static Vector3 Flat(Vector3 v)
    {
        v.y = 0f;
        return v.normalized;
    }

    /// <summary>Register an edit-mode move as a prefab-instance override, which every tile is.</summary>
    private static void RecordEditModeChange(Transform t)
    {
#if UNITY_EDITOR
        if (!Application.isPlaying && PrefabUtility.IsPartOfPrefabInstance(t))
        {
            PrefabUtility.RecordPrefabInstancePropertyModifications(t);
        }
#endif
    }

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
