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
}
