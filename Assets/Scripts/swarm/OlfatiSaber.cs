using System.Collections;
using System.Collections.Generic;
using System.Security;
using UnityEngine;
using UnityEngine.Serialization;

public class OlfatiSaber : MonoBehaviour
{
    public bool Is3D = true;

    // Unit normal of the plane the swarm is constrained to when !Is3D. Vector3.up gives the
    // horizontal (altitude-holding) formation this started as; SwarmPlaneController swings it onto
    // the pilot-steered target heading for a vertical wall. Pushed every tick by SwarmAlgorithm.
    public Vector3 PlaneNormal = Vector3.up;

    // When set, drones are pulled onto the plane at PlaneOffsetTarget along the normal — one value
    // shared by the whole swarm — rather than each toward the consensus of its own neighbours.
    // SwarmPlaneController supplies the swarm centroid's offset, so the two agree on where the plane
    // is; the shared form just states it once instead of leaving it implicit in n local averages.
    public bool HasPlaneOffsetTarget = false;
    public float PlaneOffsetTarget = 0.0f;
    public float d_ref = 7.0f;
    public float r0_coh = 150.0f;
    public float delta = 0.1f;
    public float a = 0.9f;
    public float b = 1.5f;
    public float c;
    public float gamma = 1.0f;
    // alpha-agent velocity consensus only. The beta-agent's velocity match has its own gain
    // (c2_beta) -- see there for why sharing one was a bug rather than a simplification.
    public float c_vm = 1.0f;
    public float d_obs = 5.0f;
    public float r0_obs = 6.0f;
    public float lambda_obs = 1.0f;
    public float c_obs = 4.3f;

    // beta-agent velocity-matching gain, Olfati-Saber Eq. 58's c2_beta. Deliberately NOT c_vm:
    // that is the alpha-agent velocity consensus (see GetSwarmAcceleration) and the two have no
    // reason to share a value. With c_vm = 0 -- the city scenes' deliberate choice, so that drones
    // do not match each other's velocities -- sharing it left the beta agent a purely *conservative*
    // field: a drone's approach energy came straight back out as rebound energy, which is the
    // "pushed a long way back, then takes a long time to stop" behaviour.
    // Units are s^-1: it multiplies a dimensionless bump by a world-frame velocity, so unlike
    // c_obs / d_obs it does not scale with ScaleFactor.
    public float c2_beta = 1.6f;

    // Ceiling on the obstacle force, in world m/s^2 -- the one obstacle quantity ScaleFactor does
    // *not* apply to. Applied as a smooth sigma_1 saturation rather than a hard clamp, so the field
    // never gains a kink for the attitude loop to chew on. SwarmAlgorithm further clamps this to
    // the drone's own tilt budget, since demanding more than the actuator can produce only steals
    // authority from the pilot's command without moving the drone any faster.
    public float MaxObstacleAccel = 4.0f;

    // Range of the pilot-command shield (ProjectCommandVelocity), in swarm units. Sized to the
    // tilt-limited *stopping distance*, not to d_obs: d_obs is a standoff and is deliberately much
    // tighter than the distance a drone at maxSpeed needs in order to shed that speed. At 9.31 m/s
    // and g*tan(0.436332) = 4.57 m/s^2 that distance is 9.5 m; 1.4 (14 m) leaves a drone about
    // 3 m clear of the cylinder at full stick.
    // Zero disables the shield.
    public float d_shield = 1.4f;

    // Axis of the cylinder each obstacle is approximated by (see GetObstacleFrame). World up
    // is the useful default for buildings: it makes the swarm go around them, never over.
    public Vector3 cylinderAxis = Vector3.up;
    public float ScaleFactor = 10.0f;
    // Gain of the term that pulls a drone back onto the swarming plane (formerly altitude-only).
    [FormerlySerializedAs("c_altitude_2d")]
    public float c_plane = 1.0f;

    public float MaxMigrationDistance = 10.0f;

    private string droneName;
    private VelocityControl selfVelocityControl;
    private int obstacleLayerMask;
    // Reusable overlap buffer: obstacle queries run per drone per tick, and the
    // allocating OverlapSphere would churn the GC. FixedUpdate is single-threaded,
    // so one shared buffer serves every drone.
    private static readonly Collider[] overlapBuffer = new Collider[64];

    // Frames within d_shield, kept from the last GetObstacleForce so ProjectCommandVelocity can
    // reuse them instead of running a second OverlapSphere per drone per tick. Per-instance, unlike
    // overlapBuffer, because it outlives the call. Script execution order between SwarmAlgorithm
    // and VelocityControl is undefined, so these may be one tick (20 ms, ~0.19 m at cruise) stale.
    private readonly List<ObstacleFrame> shieldFrames = new List<ObstacleFrame>();
    private bool hasWarnedBufferFull = false;

    private const string k_ObstacleLayerName = "Obstacle";

    // Awake, not Start: this component may sit disabled (SwarmAlgorithm toggles
    // the algorithm components), and GetSwarmAcceleration can be called before
    // Start would run — but Awake runs regardless of the enabled flag.
    void Awake()
    {
        selfVelocityControl = GetComponent<VelocityControl>();
        obstacleLayerMask = LayerMask.GetMask(k_ObstacleLayerName);
    }

    void Start()
    {
        droneName = transform.parent.name;
    }

    public Vector3 GetSwarmAcceleration(List<GameObject> swarm)
    {

        Vector3 velocityConsensus = Vector3.zero;
        Vector3 cohesion = Vector3.zero;
        Vector3 obstacle = Vector3.zero;

        // Get the position and velocity of the current drone
        StateFinder currentDroneState = selfVelocityControl.State;
        Vector3 position = currentDroneState.Position;
        Vector3 localVelocity = currentDroneState.VelocityVector;
        Vector3 velocity = transform.TransformDirection(localVelocity);

        // Calculate cohesion and velocity consensus for each neighbour
        float totalNeighbourPlaneOffset = 0f;
        int aliveNeighbourCount = 0;
        foreach (GameObject neighbour in swarm)
        {
            if (!SwarmRegistry.TryGet(neighbour, out SwarmRegistry.Entry entry) || entry.velocityControl == null)
                continue;

            if (entry.droneParent.gameObject == gameObject)
                continue;

            StateFinder neighbourState = entry.velocityControl.State;

            if (!neighbourState.IsAlive)
                continue;

            Vector3 neighbourPosition = neighbourState.Position;

            // Neighbour velocity in world frame
            Vector3 neighbourLocalVel = neighbourState.VelocityVector;
            Vector3 neighbourVelocity = entry.droneParent.TransformDirection(neighbourLocalVel);

            // Velocity consensus: pull toward each neighbour's velocity
            velocityConsensus += c_vm * (neighbourVelocity - velocity);

            if (!Is3D)
            {
                totalNeighbourPlaneOffset += Vector3.Dot(neighbourPosition, PlaneNormal);
                aliveNeighbourCount++;
            }

            Vector3 relativePosition = neighbourPosition - position;

            // Constrained mode: only the in-plane part of the separation drives cohesion, so the
            // formation spreads within the plane rather than around the neighbour in 3D.
            if (!Is3D)
                relativePosition -= PlaneNormal * Vector3.Dot(relativePosition, PlaneNormal);

            float distance = relativePosition.magnitude / ScaleFactor;

            // Cohesion
            cohesion += GetCohesionForce(distance, d_ref, r0_coh) * relativePosition.normalized;
        }

        // In constrained mode, correct drift off the plane. The target offset along the normal is the
        // swarm-wide one when a plane is being held (vertical mode), otherwise the mean of the
        // neighbours — which with PlaneNormal == Vector3.up is exactly the altitude-hold term this
        // grew out of.
        Vector3 planeCorrection = Vector3.zero;
        if (!Is3D && (HasPlaneOffsetTarget || aliveNeighbourCount > 0))
        {
            float targetOffset = HasPlaneOffsetTarget
                ? PlaneOffsetTarget
                : totalNeighbourPlaneOffset / aliveNeighbourCount;
            planeCorrection = c_plane * (targetOffset - Vector3.Dot(position, PlaneNormal)) * PlaneNormal;
        }

        obstacle = GetObstacleForce(position, velocity);

        return velocityConsensus + cohesion + obstacle + planeCorrection;
    }

    // Geometry of one obstacle as seen from a drone, in the cylindrical form of Olfati-Saber's
    // beta-agent: his sphere case (Eq. 57) with the sphere centre replaced by the drone's foot on
    // the cylinder axis. Shared with the debug/tuning scripts so the distance the algorithm uses
    // and the distance they report cannot drift apart.
    public struct ObstacleFrame
    {
        public bool valid;
        public float distance;   // surface distance in swarm units (already divided by ScaleFactor)
        public Vector3 outward;  // a-hat: unit radial, pointing away from the axis
        public Vector3 tangent;  // t-hat = w x a-hat: circumferential, the way around the obstacle
        public float mu;         // R / rho, clamped to 1 -> 1 at the surface, 0 far away

        // The cylinder itself. Filled in even when valid is false, so a visualiser can draw the
        // obstacle without a drone to measure from.
        public Vector3 axisCentre;
        public Vector3 axis;
        public float radius;
        public float axisHalfHeight;
    }

    // The cylinder standing in for one obstacle. Static and drone-free so the debug gizmos can
    // draw it in edit mode, where no OlfatiSaber instance has been spawned yet.
    public static void GetObstacleCylinder(Collider obstacleCollider, Vector3 cylinderAxis,
                                           out Vector3 centre, out Vector3 axis,
                                           out float radius, out float halfHeight)
    {
        axis = cylinderAxis.sqrMagnitude > 1e-8f ? cylinderAxis.normalized : Vector3.up;

        Bounds bounds = obstacleCollider.bounds;
        centre = bounds.center;

        // Smallest cylinder about that axis containing the bounds: the largest perpendicular
        // corner distance, and the box's support along the axis. Swept over the corners rather than
        // projecting the extents vector, which is only correct while the axis is aligned with one
        // of the box axes -- cylinderAxis is an inspector field and may be tilted.
        //
        // The result is conservative by construction -- root-2 times the half-width for a square
        // footprint. A tighter fit wants an explicit radius declared on the obstacle: the bounds are
        // axis-aligned and grow under rotation regardless. ObstacleCylinderGizmos draws the cylinder
        // and the bounds together so that inflation is visible.
        Vector3 e = bounds.extents;
        radius = 0.0f;
        for (int sx = -1; sx <= 1; sx += 2)
        {
            for (int sy = -1; sy <= 1; sy += 2)
            {
                for (int sz = -1; sz <= 1; sz += 2)
                {
                    Vector3 corner = new Vector3(sx * e.x, sy * e.y, sz * e.z);
                    Vector3 perpendicular = corner - axis * Vector3.Dot(corner, axis);
                    radius = Mathf.Max(radius, perpendicular.magnitude);
                }
            }
        }
        halfHeight = Mathf.Abs(e.x * axis.x) + Mathf.Abs(e.y * axis.y) + Mathf.Abs(e.z * axis.z);
    }

    // A cylinder rather than the collider bounds because a bounding box has no lateral gradient:
    // across a flat face the closest-point direction is constant, so every drone hitting that face
    // is pushed straight back the way it came and the swarm has nothing to slide along. A cylinder's
    // normal rotates continuously with the approach angle, so any off-centre approach earns a
    // tangential push that grows as the drone commits to one side.
    public static ObstacleFrame ComputeObstacleFrame(Collider obstacleCollider, Vector3 dronePosition,
                                                     Vector3 cylinderAxis, float scaleFactor)
    {
        ObstacleFrame frame = new ObstacleFrame();

        GetObstacleCylinder(obstacleCollider, cylinderAxis,
                            out frame.axisCentre, out frame.axis, out frame.radius, out frame.axisHalfHeight);

        // Foot of the drone on the axis, and the radial direction out from it.
        Vector3 fromCentre = dronePosition - frame.axisCentre;
        Vector3 radial = fromCentre - frame.axis * Vector3.Dot(fromCentre, frame.axis);
        float rho = radial.magnitude;
        if (rho < 1e-4f)
            return frame;   // sitting on the axis: no radial direction is defined

        frame.valid = true;
        frame.outward = radial / rho;
        frame.tangent = Vector3.Cross(frame.axis, frame.outward);
        frame.mu = Mathf.Min(frame.radius / rho, 1.0f);
        frame.distance = Mathf.Max(rho - frame.radius, 0.0f) / scaleFactor;
        return frame;
    }

    public ObstacleFrame GetObstacleFrame(Collider obstacleCollider, Vector3 dronePosition)
    {
        return ComputeObstacleFrame(obstacleCollider, dronePosition, cylinderAxis, ScaleFactor);
    }

    // One obstacle's two contributions, before the c_obs / c2_beta gains. Split out so the gizmos
    // can draw the repulsion and the velocity-match terms separately without restating the formula.
    public void GetObstacleContribution(ObstacleFrame frame, Vector3 droneVelocity,
                                        out Vector3 repulsion, out Vector3 velocityMatch)
    {
        // Beta-agent velocity p-hat: the radial component is removed outright, the circumferential
        // one survives scaled by mu, and motion along the axis passes through. c2_beta * (p-hat - p)
        // therefore cancels exactly the approach speed. It still opposes a fraction (1 - mu) of the
        // sideways speed, but mu -> 1 at the surface, so that fades to nothing precisely where going
        // around matters -- the inverse of the old scalar form, which damped the whole velocity
        // vector hardest at the far edge of the range.
        //
        // Note the radial removal is *symmetric*: it brakes outward motion exactly as hard as
        // inward. That is deliberate and is the half that stops the rebound -- a closing-only
        // damper would leave the drone free to be flung back out of the field.
        Vector3 vel_obs = frame.axis * Vector3.Dot(droneVelocity, frame.axis)
                        + frame.mu * frame.tangent * Vector3.Dot(droneVelocity, frame.tangent);

        // phi_beta is <= 0 and -outward points at the obstacle, so the product pushes away.
        repulsion = GetObstacleRepulsion(frame.distance) * (-frame.outward);

        // Gated by the same rho_h bump as the repulsion. Ungated, this term arrived at full strength
        // the instant a drone crossed the r0_obs query radius -- a discontinuous brake of nearly
        // -c2_beta * v applied at the point of *least* danger.
        velocityMatch = GetBetaBump(frame.distance, d_obs) * (vel_obs - droneVelocity);
    }

    // Public so the debug gizmos can draw the true resulting force rather than recomputing it.
    public Vector3 GetObstacleForce(Vector3 dronePosition, Vector3 droneVelocity)
    {
        Vector3 ObsCoh = Vector3.zero;
        Vector3 ObsVel = Vector3.zero;

        shieldFrames.Clear();

        int obstacleCount = Physics.OverlapSphereNonAlloc(dronePosition, r0_obs * ScaleFactor, overlapBuffer, obstacleLayerMask);
        if (obstacleCount == overlapBuffer.Length && !hasWarnedBufferFull)
        {
            // Silent truncation would drop obstacles in unspecified order -- possibly the nearest.
            // Once per drone: this is a 50 Hz path and the condition, once true, tends to stay true.
            hasWarnedBufferFull = true;
            Debug.LogWarning($"[OlfatiSaber] {droneName}: obstacle buffer full ({obstacleCount}); " +
                             "some obstacles are being ignored. Lower r0_obs or grow overlapBuffer.");
        }

        for (int i = 0; i < obstacleCount; i++)
        {
            ObstacleFrame frame = GetObstacleFrame(overlapBuffer[i], dronePosition);
            if (!frame.valid)
                continue;

            if (d_shield > 0.0f && frame.distance < d_shield)
                shieldFrames.Add(frame);

            GetObstacleContribution(frame, droneVelocity, out Vector3 repulsion, out Vector3 velocityMatch);
            ObsCoh += repulsion;
            ObsVel += velocityMatch;
        }

        Vector3 force = c_obs * ObsCoh + c2_beta * ObsVel;

        // Smooth saturation at MaxObstacleAccel, the vector form of the sigma_1 the paper already
        // uses inside phi_beta: unit gain near zero, 0.707 * A at |force| = A, asymptotic to A.
        // A hard clamp would work too, but this is C-infinity and its taper *below* the ceiling is
        // itself part of keeping the reaction minimal. Without it the peak repulsion exceeded the
        // tilt budget, so the excess merely crowded out the pilot's command.
        if (MaxObstacleAccel > 0.0f)
            force /= Mathf.Sqrt(1.0f + force.sqrMagnitude / (MaxObstacleAccel * MaxObstacleAccel));

        return force;
    }

    /// <summary>
    /// Removes the component of a world-frame *commanded* velocity that heads into a nearby
    /// obstacle, leaving the tangential and outward components untouched.
    /// </summary>
    /// <remarks>
    /// This is the beta-agent's own projection applied to the stick rather than to the drone.
    /// It exists because the repulsion force alone cannot win the argument: VelocityControl clamps
    /// the *sum* of the pilot and swarm accelerations to the tilt budget, while the pilot's
    /// velocity P-loop ahead of that clamp is unbounded -- full stick at a wall demands several
    /// times the actuator limit inward, so any survivable repulsion is simply outvoted.
    ///
    /// Removing only the inward component means the pilot can always fly *around* an obstacle and
    /// always fly *away* from one; only flying straight in is denied, and that fades in smoothly
    /// with the same rho_h bump the force uses. Its range is d_shield, not d_obs, because the
    /// command has to be neutralised over the stopping distance while the standoff stays tight.
    ///
    /// Note this shields the pilot's command only. Cohesion is not projected, so a drone squeezed
    /// between the formation and a facade still relies on the beta force alone.
    /// </remarks>
    public Vector3 ProjectCommandVelocity(Vector3 worldCommand)
    {
        if (d_shield <= 0.0f)
            return worldCommand;

        for (int i = 0; i < shieldFrames.Count; i++)
        {
            ObstacleFrame frame = shieldFrames[i];
            float inward = -Vector3.Dot(worldCommand, frame.outward);
            if (inward > 0.0f)
                worldCommand += GetBetaBump(frame.distance, d_shield) * inward * frame.outward;
        }

        return worldCommand;
    }

    public float GetCohesionForce(float r, float ref_d = -1, float r0 = -1)
    {
        // Use default values if parameters are not provided
        if (ref_d == -1) ref_d = d_ref;
        if (r0 == -1) r0 = r0_coh;

        float neighbourWeightDerivative = GetNeighbourWeightDerivative(r, r0);
        float cohesionIntensity = GetCohesionIntensity(r, ref_d);
        float neighbourWeight = GetNeighbourWeight(r, r0);
        float cohesionIntensityDerivative = GetCohesionIntensityDerivative(r, ref_d);

        return 1 / r0 * neighbourWeightDerivative * cohesionIntensity + neighbourWeight * cohesionIntensityDerivative;
    }

    // σ_1 saturation used by the paper's action functions: σ_1(z) = z / √(1 + z²)
    private float Sigma1(float z) => z / Mathf.Sqrt(1.0f + z * z);

    // ρ_h, Olfati-Saber Eq. 54, in the paper's own form. GetNeighbourWeight *squares* this, which
    // flattens the shoulder badly: at r = 0.9·d_obs the squared form is 26x smaller, so the outer
    // fifth of the obstacle field is effectively dead and the force appears to spring out of
    // nowhere near the surface. Kept as a separate function rather than fixing GetNeighbourWeight
    // because cohesion pairs that with GetNeighbourWeightDerivative, the analytic derivative of the
    // squared form -- changing one without the other would break the cohesion gradient, and
    // changing both would re-tune the formation, which is a different question.
    public float GetBetaBump(float r, float d)
    {
        // A zero range is "switched off", not a division by zero: r/0 is NaN at r = 0, and NaN
        // fails every comparison below, so the cos branch would return NaN into the force sum.
        if (d <= 0.0f)
            return 0.0f;

        float ratio = r / d;

        if (ratio < delta)
            return 1.0f;
        if (ratio >= 1.0f)
            return 0.0f;

        return 0.5f * (1.0f + Mathf.Cos(Mathf.PI * (ratio - delta) / (1.0f - delta)));
    }

    // Strictly-repulsive β-agent action function φ_β (Olfati-Saber Eq. 56):
    //   φ_β(r) = ρ_h(r / d_obs) · (σ_1(r − d_obs) − 1)
    // Always ≤ 0 (pushes the drone away from the obstacle) and exactly 0 for r ≥ d_obs.
    public float GetObstacleRepulsion(float r)
    {
        return GetBetaBump(r, d_obs) * (Sigma1(r - d_obs) - 1.0f);
    }

    // Cohesion intensity function
    public float GetCohesionIntensity(float r, float ref_d=-1)
    {
        // Use default value if ref_d is not provided
        if (ref_d == -1) ref_d = d_ref;

        float diff = r - ref_d;
        return ((a + b) / 2) * (Mathf.Sqrt(1 + Mathf.Pow(diff + c, 2)) - Mathf.Sqrt(1 + c * c)) + ((a - b) * diff) / 2;
    }

    // Derivative of cohesion intensity function
    float GetCohesionIntensityDerivative(float r, float ref_d=-1)
    {
        // Use default value if ref_d is not provided
        if (ref_d == -1) ref_d = d_ref;

        // Derivative of the cohesion intensity function
        float diff = r - ref_d;
        return ((a + b) / 2) * (diff + c) / Mathf.Sqrt(1 + Mathf.Pow(diff + c, 2)) + (a - b) / 2;
    }
    
    // Neighbor weight function
    public float GetNeighbourWeight(float r, float r0=-1)
    {

        // Use default value if r0 is not provided
        if (r0 == -1) r0 = r0_coh;

        float r_ratio = r / r0;

        if (r_ratio < delta)
        {
            return 1.0f;
        }
        else if (r_ratio < 1.0f)
        {
            float arg = Mathf.PI * (r_ratio - delta) / (1 - delta);
            return Mathf.Pow(0.5f * (1.0f + Mathf.Cos(arg)), 2);
        }
        else
        {
            return 0.0f;
        }
    }

    // Derivative of neighbor weight function
    float GetNeighbourWeightDerivative(float r, float r0=-1)
    {
        // Use default value if r0 is not provided
        if (r0 == -1) r0 = r0_coh;

        float r_ratio = r / r0;

        if (r_ratio < delta)
        {
            return 0.0f;
        }
        else if (r_ratio < 1.0f)
        {
            float arg = Mathf.PI * (r_ratio - delta) / (1 - delta);
            return 0.5f*(-Mathf.PI) / (1 - delta) * (1 + Mathf.Cos(arg)) * Mathf.Sin(arg);
        }
        else
        {
            return 0.0f;
        }
    }
}