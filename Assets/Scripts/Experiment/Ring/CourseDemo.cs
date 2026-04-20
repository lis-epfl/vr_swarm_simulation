using UnityEngine;
using UnityEngine.Splines;

namespace Experiment
{
    /// <summary>
    /// Animates a demo dot along the generated course spline.
    /// OnEnable initializes gates and the path but does NOT move the dot.
    /// Call BeginMoving() (via FSM) to actually start the dot animation.
    /// </summary>
    public class CourseDemo : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private CoursePathVisual pathVisual;
        [SerializeField] private RingGateManager gateManager;
        [SerializeField] private Material dotMaterial;

        [Header("Tuning")]
        [SerializeField] public float demoSpeed = 10f;
        [SerializeField] private float dotRadius = 2f;  // Large enough to reliably cross gate trigger planes

        private GameObject _demoDot;
        private float _distanceAlongSpline;
        private float _splineLength;
        private bool _initialized;
        private bool _moving;

        /// <summary>Transform of the DemoDot parent — assign this as the DemoFollowCamera target.</summary>
        public Transform DotTransform => transform;

        // ── Lifecycle ──────────────────────────────────────────────────────────

        private void OnEnable()
        {
            // Guard: if the spline isn't built yet (e.g. early enable before course generation),
            // do nothing — the FSM will call InitDemo() explicitly after generating the course.
            if (pathVisual == null || pathVisual.Spline == null || pathVisual.Spline.Spline.Count == 0)
            {
                Debug.Log("[CourseDemo] OnEnable fired but spline not ready yet — skipping. InitDemo() must be called manually.");
                return;
            }
            InitDemo();
        }

        private void OnDisable() => StopDemo();

        // ── Public API ─────────────────────────────────────────────────────────

        /// <summary>Initialize demo: show path and gate visuals, position dot at start. Does not move yet.</summary>
        public void InitDemo()
        {
            if (pathVisual == null || pathVisual.Spline == null)
            {
                Debug.LogError("[CourseDemo] pathVisual or its Spline is not assigned.");
                return;
            }

            pathVisual.PathVisible = true;

            if (_demoDot == null)
                CreateDemoDot();

            // Reset position to start of spline
            _distanceAlongSpline = 0f;
            _splineLength = pathVisual.Spline.CalculateLength();
            PositionDotAtDistance(0f);
            _demoDot.SetActive(true);

            // Init gate visuals without starting the timer or marking course as running
            if (gateManager != null)
                gateManager.StartDemoMode();

            _initialized = true;
            _moving = false;
            Debug.Log($"[CourseDemo] Initialized. Spline length: {_splineLength:F1}m. Waiting for BeginMoving().");
        }

        /// <summary>Start moving the dot. Call this when the operator is ready to show the live demo.</summary>
        public void BeginMoving()
        {
            if (!_initialized)
            {
                Debug.LogWarning("[CourseDemo] BeginMoving() called before InitDemo(). Calling InitDemo() first.");
                InitDemo();
            }
            _moving = true;
            Debug.Log("[CourseDemo] Dot is now moving.");
        }

        /// <summary>Stop everything and hide the demo.</summary>
        public void StopDemo()
        {
            _moving = false;
            _initialized = false;

            if (_demoDot != null)
                _demoDot.SetActive(false);

            if (pathVisual != null)
                pathVisual.PathVisible = false;

            if (gateManager != null)
                gateManager.DemoMode = false;

            Debug.Log("[CourseDemo] Stopped.");
        }

        // ── Update ─────────────────────────────────────────────────────────────

        private void Update()
        {
            if (!_moving || _splineLength <= 0f)
                return;

            _distanceAlongSpline += demoSpeed * Time.deltaTime;
            _distanceAlongSpline %= _splineLength;

            PositionDotAtDistance(_distanceAlongSpline);
        }

        // ── Helpers ────────────────────────────────────────────────────────────

        private void PositionDotAtDistance(float distance)
        {
            if (pathVisual?.Spline == null) return;

            float t = pathVisual.Spline.Spline.ConvertIndexUnit(
                distance, PathIndexUnit.Distance, PathIndexUnit.Normalized);

            Vector3 pos     = pathVisual.Spline.EvaluatePosition(t);
            Vector3 tangent = pathVisual.Spline.EvaluateTangent(t);

            // Move the parent transform — DemoFollowCamera tracks this
            transform.position = pos;
            if (tangent.sqrMagnitude > 0.001f)
                transform.forward = tangent.normalized;
        }

        private void CreateDemoDot()
        {
            _demoDot = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _demoDot.name = "DemoDot_Sphere";
            _demoDot.transform.SetParent(transform);
            _demoDot.transform.localPosition = Vector3.zero;
            _demoDot.transform.localScale    = Vector3.one * dotRadius * 2f;

            // Tag as Player so RingGate recognizes it
            _demoDot.tag = "Player";

            // Rigidbody required for Unity trigger callbacks to fire
            Rigidbody rb = _demoDot.AddComponent<Rigidbody>();
            rb.isKinematic = true;   // We move it manually; physics should not simulate it
            rb.useGravity  = false;

            // Keep the sphere collider (non-trigger) so it enters gate trigger volumes
            SphereCollider col = _demoDot.GetComponent<SphereCollider>();
            if (col != null)
                col.isTrigger = false;

            MeshRenderer renderer = _demoDot.GetComponent<MeshRenderer>();
            if (renderer != null && dotMaterial != null)
                renderer.material = dotMaterial;

            _demoDot.SetActive(false);
            Debug.Log($"[CourseDemo] Created demo dot (radius={dotRadius}, tag=Player, Rigidbody kinematic).");
        }
    }
}
