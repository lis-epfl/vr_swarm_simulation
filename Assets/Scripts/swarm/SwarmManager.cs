using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SwarmManager : MonoBehaviour
{
    public static SwarmManager Instance;

    public enum SwarmAlgorithm
    {
        NONE,
        REYNOLDS,
        OLFATI_SABER,
    }

    [Header("Swarm Algorithm")]
    public SwarmAlgorithm swarmAlgorithm;
    public bool is3D = true;

    [Header("Reynolds Parameters")]
    public float cohesionWeight = 1.0f;
    public float separationWeight = 15.0f;
    public float alignmentWeight = 1.0f;

    [Header("Olfati-Saber Parameters")]
    public float d_ref = 7.0f;
    public float r0_coh = 20.0f;
    [Tooltip("Flat-top fraction of the rho_h bump, and one value for every bump there is: the alpha " +
             "neighbour weight (over r0_coh), the obstacle field (over d_obs), the pilot shield's fade " +
             "(over d_shield) and the hollow core. Lowering it fades attraction beyond delta*r0_coh, and " +
             "weakens the obstacle field and the shield through the middle of their ranges, so both " +
             "build up more gradually as a drone closes in.")]
    public float delta = 0.1f;
    [Tooltip("Attraction per neighbour at long range, m/s^2. With r0_coh far larger than the swarm every " +
             "neighbour pulls, so a drone the formation leaves behind a building is dragged through it by " +
             "the whole swarm while the obstacle force saturates at maxObstacleAccel -- that was nearly " +
             "every building contact in ScaledCityWorld.\n\n" +
             "It is also the formation's stiffness, and the two cannot be separated: the sigma_1 shape " +
             "ties the slope of phi near d_ref to this asymptote (b moves the ratio by ~5% at most). So " +
             "lowering it slows how fast the swarm follows the spread stick -- mostly contraction, which " +
             "only attraction drives. It is the one dial between crash safety and spread response.")]
    public float a = 0.3f;
    public float b = 0.5f;
    private float c;
    public float gamma = 1.0f;
    [Tooltip("Alpha-agent velocity consensus gain only. The obstacle (beta-agent) velocity match " +
             "has its own gain, c2_beta -- setting this to 0 no longer disables obstacle damping.\n\n" +
             "Summed over every alive neighbour with no distance weighting, so a drone feels (N-1) times " +
             "this, and it hands a drone the pilot shield is braking back the command the shield just " +
             "removed. Near buildings even a few hundredths trades building contacts for fewer " +
             "drone-drone collisions, and it damps the formation's response to the spread stick.")]
    public float c_vm = 1.0f;
    public float d_obs = 4.0f;
    public float r0_obs = 6.0f;
    public float lambda_obs = 1.0f;
    public float c_obs = 4.3f;
    [Tooltip("Beta-agent velocity-matching gain (Olfati-Saber c2_beta), s^-1. This is what stops a " +
             "drone rebounding off an obstacle: without it the obstacle field is conservative and " +
             "returns all the approach energy. Roughly critical at 2*sqrt(peak accel / field depth).")]
    public float c2_beta = 1.6f;
    [Tooltip("Ceiling on the obstacle force in m/s^2 (world units, not scaled by scaleFactor). " +
             "Further clamped per drone to its own tilt budget, g*tan(min(maxPitch, maxRoll)).")]
    public float maxObstacleAccel = 4.0f;
    [Tooltip("Range of the pilot-command shield in swarm units (metres / scaleFactor). Inward stick " +
             "is faded out over this distance so the pilot cannot fly straight into an obstacle. " +
             "Size it to the stopping distance, which is much larger than the d_obs standoff. 0 = off.\n\n" +
             "That is one drone's floor; a swarm wants more. The fade also decides how differently " +
             "neighbours brake beside a building, and an abrupt one builds the closing speed behind most " +
             "drone-drone collisions there.")]
    public float d_shield = 1.4f;
    public float scaleFactor = 10.0f;

    [Header("Hollow Swarm Core")]
    [Tooltip("Hollow the middle of the swarm so more drones sit on the convex hull and therefore " +
             "have their camera feed shown to the pilot. Works by standing a virtual Olfati-Saber " +
             "beta-agent (a cylinder) at the swarm centroid, so being in the middle costs energy " +
             "and the equilibrium becomes a ring. No formation, no assigned slots: the drones stay " +
             "interchangeable and the swarm still deforms freely around obstacles.\n\n" +
             "This is the single switch for the whole feature. Unticked, every field below is inert " +
             "and the swarm behaves exactly as it did before the feature existed.")]
    public bool hollowSwarmCore = false;

    [Tooltip("Core radius as a fraction of how big the formation measurably is (its mean distance " +
             "from the centroid). 0.5 is the recommended value; 0.4-0.6 all put every drone on the " +
             "hull.\n\n" +
             "It MUST stay well under 1: a beta-agent only has a gradient outside its cylinder, so " +
             "a core sized to the ring the drones should end up on swallows the whole swarm in its " +
             "flat interior, where the result is either no effect at all or the swarm escaping " +
             "outright. Measured rather than predicted from d_ref because the swarm's equilibrium " +
             "spacing is about half of d_ref, so a radius derived from the commanded spacing lands " +
             "roughly twice as far out as the drones ever go.")]
    public float coreRadiusFraction = 0.5f;

    [Tooltip("Time constant (s) of the low-pass on the core radius. It exists for drone losses and " +
             "for the mild feedback in measuring the core off the swarm it is shaping. Seeded on " +
             "the first tick, so there is no ramp at scene start or on leaving plane mode.\n\n" +
             "Its lag is felt when the pilot pulls the spread in: the core is still sized to the old, " +
             "wider ring and pushes outward against the contraction. 0.5 s cut the settling time after " +
             "a spread change by about a third with no change to the ring or to crashes; " +
             "the loop gain is low enough (see SwarmPlaneController.UpdateCoreRadius) that a fast " +
             "filter does not oscillate.")]
    public float coreRadiusFilterTime = 2.0f;

    [Tooltip("Core repulsion gain. 1.5 is verified to put every drone on the hull at 6 and 10 " +
             "drones, from any starting spread, and is insensitive to the cohesion range.")]
    public float c_core = 1.5f;

    [Tooltip("OPTIONAL, off by default. Cohesion interaction range as a multiple of the live d_ref " +
             "(which the spread stick rewrites every tick, so a fixed r0_coh changes the shape of " +
             "the well under the pilot's hand). 0 = off, use r0_coh verbatim.\n\n" +
             "Kept as a knob, but relaxing the coded force law says it does NOT help the hull " +
             "count: from the current 18.5 down to 5 the equilibrium is unchanged (9 of 10 on the " +
             "hull, spacing 0.50 d_ref), at 4 it gets worse, and at 3 the swarm disperses. The core " +
             "above does the whole job on its own and is unaffected by this. Clamped to a floor of " +
             "3 in OnValidate. NOTE this is not the paper's ~1.2 -- see OlfatiSaber.r0CohRatio.")]
    public float r0CohRatio = 0.0f;

    [Tooltip("Core beta-agent velocity-match gain (s^-1). Damps radial overshoot without touching " +
             "travel around the ring. Set to 0 and the core becomes conservative: drones pushed out " +
             "spring back in and the ring breathes.\n\n" +
             "It is measured against each drone's absolute velocity while the core travels with the " +
             "swarm, so at cruise it is also a drag on the whole formation (about 0.7 m/s^2 at 1.6) -- " +
             "one that fades out beside buildings, i.e. differs between neighbours exactly there.")]
    public float c2_core = 1.6f;

    [Tooltip("Core standoff as a multiple of the live d_ref, in swarm units. Sized to the lattice " +
             "spacing so the annulus comes out about one cell thick -- deliberately not d_obs, " +
             "which is sized to the surface of a building.")]
    public float coreStandoffRatio = 0.5f;

    [Tooltip("Ceiling on the core force in m/s^2. Keep it well under maxObstacleAccel: that gap is " +
             "what guarantees a building wins where the two disagree, and the swarm can still " +
             "deform freely around obstacles.")]
    public float maxCoreAccel = 2.0f;

    public enum AttitudeAlgorithm
    {
        NONE,
        SIMPLE,
        // Boundary detected from the local hull of each drone's NumNeighbours nearest neighbours
        // (cheap, but flags interior drones as boundary because the local point set is tiny).
        LOCAL_CONVEXHULL,
        // Boundary detected from the convex hull of the whole swarm (only true outer-ring drones).
        GLOBAL_CONVEXHULL,
    }

    [Header("Altitude Limit")]
    [Tooltip("Metres above the TERRAIN surface that a drone may be commanded to. 0 (or a scene " +
             "with no Terrain, e.g. DJIScene and FactoryScene) = no limit. The height measured " +
             "from is the Unity Terrain under the drone, not the rooftops, so the ceiling stays " +
             "flat over the city: at 50 in ScaledCityWorld, whose buildings are all exactly 50 " +
             "tall, the swarm must fly around a building rather than over it. This is a hard cap " +
             "on the height SETPOINT (the symmetric counterpart of VelocityControl.MinHeight), so " +
             "the climb stick simply stops having effect upward and nothing winds up; the swarm's " +
             "own upward force is dropped above the ceiling as well, since in a horizontal " +
             "formation it reaches thrust directly rather than through the setpoint. Per scene, " +
             "because the value is a property of the scene's geometry: the city pack's unscaled " +
             "buildings reach ~136 m, so CityWorld and CrowdWorld need far more headroom than " +
             "ScaledCityWorld's uniform 50 m skyline. NOTE the height loop holds each drone about " +
             "1.3 m above its setpoint (see the ForceMode note in CLAUDE.md), so the flown ceiling " +
             "is about 1.3 m above this number.")]
    public float maxHeightAboveTerrain = 50.0f;

    [Header("Attitude Control")]
    public AttitudeAlgorithm SelectedAttitudeAlgorithm;
    public int numNeighbours = 5;
    public int numDimensions = 2;
    public bool pointInwards = true;
    [Tooltip("A drone that has just joined the swarm boundary keeps its feed screen hidden until its " +
             "heading is first within this many degrees of its target heading, so the pilot never " +
             "watches it swing round. Latched: once shown, the screen stays up until the drone leaves " +
             "the boundary. 180 disables the wait.")]
    [Range(0.0f, 180.0f)]
    public float feedHeadingToleranceDeg = 5.0f;

    [Header("Look-Direction Gap Fill")]
    [Tooltip("Turn the two hull drones either side of the pilot's body yaw slightly towards it when " +
             "neither is facing that way. The hull rule points every boundary drone along its outward " +
             "vertex bisector, which leaves a blind spot exactly where the pilot is flying when the " +
             "swarm splits around a building: the front drone is stopped and swallowed, its two " +
             "neighbours pass either side ahead of the lagging swarm, and the sharp hull corners they " +
             "form point them 40-50 deg off the flight direction.\n\n" +
             "Only that bracketing pair moves, by at most lookGapMaxShiftDeg, and never past the look " +
             "direction. Acts in GLOBAL_CONVEXHULL only, and not in vertical-plane mode (which has its " +
             "own shared heading) or with the gimbal at or below FPVCameraScript.NadirPitch (a camera " +
             "looking down has no direction to fill). Unticked, every field below is inert and the hull " +
             "rule is exactly what it was.\n\n" +
             "On ScaledCityWorld's 120 held-out flights (headless, pilot facing the way they fly) it cut " +
             "the time the look direction sat more than 30 deg from every shown drone's heading from 8.8% " +
             "to 1.1%, and the time it sat outside every shown image from 0.4% to zero, with drones lost " +
             "and building contacts unchanged.")]
    public bool fillLookDirectionGap = false;

    [Tooltip("The look direction counts as covered, and nothing moves, while a hull drone's target " +
             "heading is within this many degrees of the pilot's body yaw. Beyond it the pair turns one " +
             "degree per degree, up to lookGapMaxShiftDeg.\n\n" +
             "An evenly spread ring of 9-10 hull drones puts the outward headings 36-40 deg apart, so at 20 " +
             "the rule is idle there. A flying swarm is not that ring: in ScaledCityWorld's held-out " +
             "flights about 7.7 of 10 drones are on the hull, and the fill turns a pair about a quarter of " +
             "the time, by 7 deg on average. Raise this to keep it for the large gaps only.")]
    [Range(0.0f, 90.0f)]
    public float lookGapCoverageDeg = 20.0f;

    [Tooltip("The most either drone of the bracketing pair may be turned off its hull heading. The " +
             "FPV camera sees about 98 deg horizontally, so a front pair at +/-50 deg leaves no overlap " +
             "at all straight ahead; 15 takes it to +/-35 deg, about 28 deg of overlap. The price is " +
             "paid beside the pair: the gaps to their outer neighbours widen by the same amount.")]
    [Range(0.0f, 45.0f)]
    public float lookGapMaxShiftDeg = 15.0f;

    [Header("Camera Gimbal")]
    [Tooltip("Swarm-wide FPV camera gimbal pitch in degrees (DJI convention): 0 = level " +
             "horizon, negative = look down (to -90 = straight down), positive = look up " +
             "(to +60). Changing this at runtime tilts every drone's camera.")]
    [Range(FPVCameraScript.MinPitch, FPVCameraScript.MaxPitch)]
    public float gimbalPitch = 0f;


    public delegate void OnSwarmParamsChanged();
    public event OnSwarmParamsChanged swarmParamsChanged;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);

            // The swarming plane is toggled at runtime and read by every drone, so it must exist
            // wherever a swarm does. Attach it here rather than making every scene remember to.
            if (SwarmPlaneController.Instance == null && GetComponent<SwarmPlaneController>() == null)
            {
                gameObject.AddComponent<SwarmPlaneController>();
            }
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        // Push the configured gimbal pitch to every drone's FPV camera at startup.
        ApplyGimbalPitch();
    }

    // Called whenever a value is changed in the Inspector
    private void OnValidate()
    {
        // Below ~3 the cohesion well is too shallow to hold the swarm together (see the tooltip),
        // and with c_vm = 0 there is no velocity consensus to catch a drone that falls out of range.
        // Clamped here rather than trusted to the inspector because the failure is irrecoverable.
        if (r0CohRatio > 0.0f) r0CohRatio = Mathf.Max(r0CohRatio, 3.0f);

        // Trigger the event to notify all subscribed drones
        swarmParamsChanged?.Invoke();

        // Apply the gimbal pitch live so it can be tuned during runtime without selecting a drone.
        ApplyGimbalPitch();
    }

    // Drives the swarm-wide FPV gimbal pitch (shared by every drone).
    private void ApplyGimbalPitch()
    {
        FPVCameraScript.SetPitch(gimbalPitch);
    }

    // Drives the swarm-wide FPV gimbal pitch from a normalized dial input in [-1, 1]
    // (e.g. the joystick pitch dial), mapping linearly across the full gimbal range
    // (dial -1 = straight down, +1 = up). Updates the Inspector field so the value is
    // visible/tunable, then pushes it to every drone's camera.
    public void SetGimbalPitchNormalized(float dial)
    {
        dial = Mathf.Clamp(dial, -1f, 1f);
        gimbalPitch = Mathf.Lerp(FPVCameraScript.MinPitch, FPVCameraScript.MaxPitch,
                                 (dial + 1f) * 0.5f);
        ApplyGimbalPitch();
    }

    // Getters
    public bool GetDimensions() => is3D;

    // Getters for the Reynolds parameters
    public float GetCohesionWeight() => cohesionWeight;
    public float GetSeparationWeight() => separationWeight;
    public float GetAlignmentWeight() => alignmentWeight;

    // Getters for the Olfati-Saber parameters
    public float GetDRef() => d_ref;
    // Setter so the keyboard/joystick spread command is reflected in the inspector.
    public void SetDRef(float value) => d_ref = value;
    public float GetR0Coh() => r0_coh;

    /// <summary>
    /// The cohesion interaction range actually in force, in swarm units: a multiple of the live
    /// d_ref while the hollow-core feature is on, the raw r0_coh otherwise. Mirrors
    /// <see cref="OlfatiSaber.EffectiveR0Coh"/>, and exists so that consumers reasoning about how
    /// far away is "still attached to the swarm" — DroneHealthMonitor — cannot go on using a range
    /// the swarm itself has stopped using.
    /// </summary>
    public float GetEffectiveR0Coh() => (hollowSwarmCore && r0CohRatio > 0.0f) ? r0CohRatio * d_ref : r0_coh;
    public float GetDelta() => delta;
    public float GetA() => a;
    public float GetB() => b;
    public float GetC() => (b - a) / (2 * Mathf.Sqrt(a * b));
    public float GetGamma() => gamma;
    public float GetCVM() => c_vm;
    public float GetDObs() => d_obs;
    public float GetR0Obs() => r0_obs;
    public float GetLambdaObs() => lambda_obs;
    public float GetCObs() => c_obs;
    public float GetC2Beta() => c2_beta;
    public float GetMaxObstacleAccel() => maxObstacleAccel;
    public float GetDShield() => d_shield;
    public float GetScaleFactor() => scaleFactor;

    // Getters for the hollow swarm core
    public bool GetHollowSwarmCore() => hollowSwarmCore;
    public float GetR0CohRatio() => r0CohRatio;
    public float GetCoreRadiusFraction() => coreRadiusFraction;
    public float GetCoreRadiusFilterTime() => coreRadiusFilterTime;
    public float GetCCore() => c_core;
    public float GetC2Core() => c2_core;
    public float GetCoreStandoffRatio() => coreStandoffRatio;
    public float GetMaxCoreAccel() => maxCoreAccel;

    // Getter for the altitude ceiling
    public float GetMaxHeightAboveTerrain() => maxHeightAboveTerrain;

    // Getters for the attitude control
    public int GetNumNeighbours() => numNeighbours;
    public int GetNumDimensions() => numDimensions;
    public bool GetPointInwards() => pointInwards;
    public float GetFeedHeadingToleranceDeg() => feedHeadingToleranceDeg;
    public AttitudeAlgorithm GetSelectedAttitudeAlgorithm() => SelectedAttitudeAlgorithm;

    // Getters for the look-direction gap fill
    public bool GetFillLookDirectionGap() => fillLookDirectionGap;
    public float GetLookGapCoverageDeg() => lookGapCoverageDeg;
    public float GetLookGapMaxShiftDeg() => lookGapMaxShiftDeg;

    // Getter for the swarm-wide FPV camera gimbal pitch
    public float GetGimbalPitch() => gimbalPitch;

}
