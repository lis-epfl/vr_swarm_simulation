using System.Collections.Generic;
using UnityEngine;

// Cache of each drone's per-tick components, keyed by the drone root ("Drone N")
// GameObject. The swarm hot loops (OlfatiSaber / Reynolds / AttitudeAlgorithm)
// run per neighbour per FixedUpdate, so resolving "DroneParent" by string search
// and GetComponent there is O(n²) per tick — this registry resolves each drone
// once. Unregistered drones are resolved lazily on first lookup, so scenes that
// don't spawn through swarmSpawn keep the old Find semantics without the cost.
public static class SwarmRegistry
{
    public struct Entry
    {
        public Transform droneParent;
        public VelocityControl velocityControl;
        public AttitudeAlgorithm attitude;
        public Rigidbody rigidbody;
    }

    private static readonly Dictionary<GameObject, Entry> entries = new Dictionary<GameObject, Entry>();

    // Statics survive entering play mode when domain reload is disabled; start
    // each run with a clean table so stale destroyed-object entries never leak in.
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetOnLoad()
    {
        entries.Clear();
    }

    public static void Register(GameObject droneRoot)
    {
        if (droneRoot == null) return;
        Transform droneParent = droneRoot.transform.Find("DroneParent");
        if (droneParent == null) return;
        entries[droneRoot] = new Entry
        {
            droneParent = droneParent,
            velocityControl = droneParent.GetComponent<VelocityControl>(),
            attitude = droneParent.GetComponent<AttitudeAlgorithm>(),
            rigidbody = droneParent.GetComponent<Rigidbody>(),
        };
    }

    public static bool TryGet(GameObject droneRoot, out Entry entry)
    {
        if (droneRoot == null)
        {
            entry = default;
            return false;
        }
        if (entries.TryGetValue(droneRoot, out entry)) return true;
        Register(droneRoot);
        return entries.TryGetValue(droneRoot, out entry);
    }

    /// <summary>
    /// The one alive drone, when exactly one is left — only one was spawned, or every other one
    /// has crashed. False for zero or for two or more. A single drone has no swarm to take a
    /// heading from, so AttitudeAlgorithm hands it the yaw stick and PyUniSharingFast slaves the
    /// body yaw to its commanded heading. Destroyed drones are skipped; a drone with no
    /// VelocityControl counts as alive, as in PyUniSharingFast.IsAlive.
    /// </summary>
    public static bool TryGetLoneDrone(out GameObject droneRoot, out Entry entry)
    {
        droneRoot = null;
        entry = default;
        int alive = 0;
        foreach (KeyValuePair<GameObject, Entry> pair in entries)
        {
            if (pair.Key == null) continue;
            VelocityControl vc = pair.Value.velocityControl;
            if (vc != null && vc.State != null && !vc.State.IsAlive) continue;
            if (++alive > 1) return false;
            droneRoot = pair.Key;
            entry = pair.Value;
        }
        if (alive == 1) return true;
        droneRoot = null;
        entry = default;
        return false;
    }
}
