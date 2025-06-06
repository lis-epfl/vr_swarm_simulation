using System.Collections.Generic;
using UnityEngine.XR.Hands.Processing;

namespace UnityEngine.XR.Hands.Samples.VisualizerSample
{
    /// <summary>
    /// Example hand processor that applies transformations on the root poses to
    /// modify the hands skeleton. Note it is possible to modify the bones
    /// directly for more advanced use cases that are not shown here.
    /// </summary>
    public class HandProcessor : MonoBehaviour, IXRHandProcessor
    {
        /// <inheritdoc />
        public int callbackOrder => 0;

        /// <summary>
        /// The mode to use for the sample processor.
        /// </summary>
        public enum ProcessorExampleMode
        {
            /// <summary>
            /// No processing is applied.
            /// </summary>
            None,

            /// <summary>
            /// Smooths the hand root pose of the left and right hands with interpolated positions
            /// </summary>
            Smoothing,

            /// <summary>
            /// Inverts the left and right hands.
            /// </summary>
            Invert,

            /// <summary>
            /// Detects shapes formed by both hands together.
            /// </summary>
            ShapeDetection /// MINE
        }

        // Variables used for smoothing hand movements.
        bool m_FirstFrame = false;
        Vector3 m_LastLeftHandPosition;
        Vector3 m_LastRightHandPosition;
        Pose m_LeftHandPose = Pose.identity;
        Pose m_RightHandPose = Pose.identity;

        [SerializeField]
        [Tooltip("The mode to use for the sample processor.")]
        ProcessorExampleMode m_ProcessorExampleMode = ProcessorExampleMode.Smoothing;
        ProcessorExampleMode m_LastProcessorExampleMode = ProcessorExampleMode.None;

        /// <summary>
        /// The <see cref="ProcessorExampleMode"/> to use for the sample processor.
        /// </summary>
        public ProcessorExampleMode processorExampleMode
        {
            get => m_ProcessorExampleMode;
            set => m_ProcessorExampleMode = value;
        }

        // Smoothing factors for the left and right hands.
        [Header("Smoothing parameters")]
        [SerializeField]
        [Tooltip("The smoothing factor to use when smoothing the root of the left hand in the sample processor. Use 0 for no smoothing.")]
        float m_LeftHandSmoothingFactor = 16f;

        [SerializeField]
        [Tooltip("The smoothing factor to use when smoothing the root of the right hand in the sample processor. Use 0 for no smoothing.")]
        float m_RightHandSmoothingFactor = 16f;

        [Header("Shape Detection Parameters")]
        [SerializeField]
        [Tooltip("How strict should the shape detection be")]
        LeniencyLevel m_LeniencyLevel = LeniencyLevel.Medium;
        
        [SerializeField]
        [Tooltip("Prefab for the shape visualization")]
        GameObject m_ShapePrefab;
        
        [SerializeField]
        [Tooltip("Minimum scale factor for the ellipsoid's width")]
        float m_MinWidthScale = 0.05f;
        
        [SerializeField]
        [Tooltip("Maximum scale factor for the ellipsoid's width")]
        float m_MaxWidthScale = 0.3f;
        
        [SerializeField]
        [Tooltip("Minimum acceptable dot product between palms (-1 = facing perfectly, 1 = same direction)")]
        float m_PalmDotThreshold = 0.9f; // Much more lenient threshold
        
        [SerializeField]
        [Tooltip("Default width if finger detection fails")]
        float m_DefaultWidth = 0.1f;
        
        [SerializeField]
        [Tooltip("Always show ellipsoid when both hands are tracked")]
        bool m_AlwaysShowEllipsoid = true;

        [Header("Camera Offset")]
        [SerializeField]
        [Tooltip("The GameObject that defines the camera offset. Its position will be used as the basis for the ellipsoid.")]
        private GameObject cameraOffset;

        // New enum to control leniency levels
        public enum LeniencyLevel
        {
            VeryStrict,   // Requires palms to face each other almost perfectly (dot product < -0.7)
            Strict,       // Requires palms to face each other reasonably well (dot product < -0.3)
            Medium,       // Somewhat lenient palm orientation check (dot product < 0.2)
            Lenient,      // Very relaxed palm orientation check (dot product < 0.7)
            VeryLenient   // Shows ellipsoid whenever hands are tracked (bypass all checks)
        }
        
        // Helper method to get the appropriate dot product threshold based on leniency level
        private float GetPalmDotThresholdForLeniency()
        {
            switch (m_LeniencyLevel)
            {
                case LeniencyLevel.VeryStrict: return -0.7f;   // Palms must face each other almost directly
                case LeniencyLevel.Strict: return -0.3f;       // Palms must face each other well
                case LeniencyLevel.Medium: return 0.2f;        // Palms must somewhat face each other
                case LeniencyLevel.Lenient: return 0.7f;       // Palms can be only slightly facing each other
                case LeniencyLevel.VeryLenient: return 1.0f;   // Any orientation works
                default: return 0.2f;                          // Default to Medium
            }
        }
        
        // Helper method to determine if we should bypass all checks
        private bool ShouldBypassAllChecks()
        {
            return m_LeniencyLevel == LeniencyLevel.VeryLenient;
        }

        // Runtime objects
        private GameObject m_ShapeInstance;
        private bool m_IsShapeActive = false;

        public static float HandEllipsoidLength = 0f;  // updated with palm-to-palm distance
        public static float HandEllipsoidWidth = 0f;   // updated with width scale from index fingers
        public static Vector3 LeftPalmPosition = Vector3.zero;  // ADDED: Static variable for left palm position
        public static Vector3 RightPalmPosition = Vector3.zero; // ADDED: Static variable for right palm position
        public static bool ArePalmsTracked = false; // ADDED: Flag to indicate if palms used for ellipsoid are tracked

        /// <inheritdoc />
        public void ProcessJoints(XRHandSubsystem subsystem, XRHandSubsystem.UpdateSuccessFlags successFlags, XRHandSubsystem.UpdateType updateType)
        {
            switch (m_ProcessorExampleMode)
            {
                case ProcessorExampleMode.Smoothing:
                    SmoothHandsExample(subsystem, successFlags, updateType, m_LastProcessorExampleMode != m_ProcessorExampleMode);
                    break;

                case ProcessorExampleMode.Invert:
                    InvertHandsExample(subsystem, successFlags, updateType);
                    break;

                case ProcessorExampleMode.ShapeDetection:
                    DetectHandShapes(subsystem, successFlags, updateType);
                    break;
            }

            m_LastProcessorExampleMode = m_ProcessorExampleMode;
        }

        // Smooths the hand movements of an XRHandSubsystem by updating the root
        // pose of the left and right hands with interpolated positions.
        void SmoothHandsExample(XRHandSubsystem subsystem, XRHandSubsystem.UpdateSuccessFlags successFlags, XRHandSubsystem.UpdateType updateType, bool modeChanged)
        {
            var leftHand = subsystem.leftHand;
            var rightHand = subsystem.rightHand;

            if (leftHand.isTracked && m_LeftHandSmoothingFactor > 0)
            {
                var leftPose = leftHand.rootPose;
                var currentLeftHandPosition = leftPose.position;
                if (!m_FirstFrame && !modeChanged)
                {
                    float tweenAmt = Time.deltaTime * m_LeftHandSmoothingFactor;
                    currentLeftHandPosition = Vector3.Lerp(m_LastLeftHandPosition, currentLeftHandPosition, tweenAmt);
                    m_LeftHandPose.position = currentLeftHandPosition;
                    m_LeftHandPose.rotation = leftPose.rotation;

                    leftHand.SetRootPose(m_LeftHandPose);
                    subsystem.SetCorrespondingHand(leftHand);
                }
                m_LastLeftHandPosition = currentLeftHandPosition;
            }

            if (rightHand.isTracked && m_RightHandSmoothingFactor > 0)
            {
                var rightPose = rightHand.rootPose;
                var currentRightHandPosition = rightPose.position;
                if (!m_FirstFrame && !modeChanged)
                {
                    float tweenAmt = Time.deltaTime * m_RightHandSmoothingFactor;
                    currentRightHandPosition = Vector3.Lerp(m_LastRightHandPosition, currentRightHandPosition, tweenAmt);
                    m_RightHandPose.position = currentRightHandPosition;
                    m_RightHandPose.rotation = rightPose.rotation;

                    rightHand.SetRootPose(m_RightHandPose);
                    subsystem.SetCorrespondingHand(rightHand);
                }
                m_LastRightHandPosition = currentRightHandPosition;
            }
        }

        // Call this from process joints to try inverting the user's hands.
        void InvertHandsExample(XRHandSubsystem subsystem, XRHandSubsystem.UpdateSuccessFlags successFlags, XRHandSubsystem.UpdateType updateType)
        {
            var leftHand = subsystem.leftHand;
            var leftHandPose = leftHand.rootPose;

            var rightHand = subsystem.rightHand;
            var rightHandPose = rightHand.rootPose;

            if (leftHand.isTracked)
            {
                leftHand.SetRootPose(rightHandPose);
                subsystem.SetCorrespondingHand(leftHand);

                rightHand.SetRootPose(leftHandPose);
                subsystem.SetCorrespondingHand(rightHand);
            }
        }

        void DetectHandShapes(XRHandSubsystem subsystem, XRHandSubsystem.UpdateSuccessFlags successFlags, XRHandSubsystem.UpdateType updateType)
        {
            var leftHand = subsystem.leftHand;
            var rightHand = subsystem.rightHand;
        
            // Check if both hands are tracked
            if (!leftHand.isTracked || !rightHand.isTracked)
            {
                HideShape();
                ArePalmsTracked = false; // ADDED: Reset flag
                return;
            }
        
            // Get palm positions
            if (!leftHand.GetJoint(XRHandJointID.Palm).TryGetPose(out Pose leftPalmPose) || 
                !rightHand.GetJoint(XRHandJointID.Palm).TryGetPose(out Pose rightPalmPose))
            {
                HideShape();
                ArePalmsTracked = false; // ADDED: Reset flag
                return;
            }

            // --- Palms are successfully tracked from here ---
            ArePalmsTracked = true; // ADDED: Set flag

            // ADDED: Update static palm positions
            LeftPalmPosition = leftPalmPose.position;
            RightPalmPosition = rightPalmPose.position;
            
            // Calculate palm-to-palm vector and distance early
            Vector3 palmTopalm = RightPalmPosition - LeftPalmPosition; // Use updated static vars
            float palmDistance = palmTopalm.magnitude;
            Vector3 mainAxis = palmTopalm.normalized;
            
            // Check if palms are facing each other (based on leniency level)
            // Skip this check at the highest leniency level
            if (!ShouldBypassAllChecks())
            {
                float palmDotProduct = Vector3.Dot(leftPalmPose.forward, rightPalmPose.forward);
                float dotThreshold = GetPalmDotThresholdForLeniency();
                
                if (palmDotProduct > dotThreshold)
                {
                    HideShape();
                    return;
                }
                
                // For stricter levels, also check palm distance
                if (m_LeniencyLevel == LeniencyLevel.VeryStrict || m_LeniencyLevel == LeniencyLevel.Strict)
                {
                    float maxDistance = m_LeniencyLevel == LeniencyLevel.VeryStrict ? 0.4f : 0.6f;
                    
                    if (palmDistance > maxDistance)
                    {
                        HideShape();
                        return;
                    }
                }
            }
            
            // Calculate center point between palms
            Vector3 centerPoint = (leftPalmPose.position + rightPalmPose.position) * 0.5f;
            
            // Default width in case finger detection fails
            float widthScale = m_DefaultWidth;
            
            // Try to get index finger positions for better width calculation
            bool hasLeftIndex = leftHand.GetJoint(XRHandJointID.IndexTip).TryGetPose(out Pose leftIndexTip);
            bool hasRightIndex = rightHand.GetJoint(XRHandJointID.IndexTip).TryGetPose(out Pose rightIndexTip);
            
            // If we have finger positions, calculate width based on them
            if (hasLeftIndex && hasRightIndex)
            {
                // Calculate distances of index fingers from the main axis
                float leftIndexDistance = Vector3.Cross(leftIndexTip.position - leftPalmPose.position, mainAxis).magnitude;
                float rightIndexDistance = Vector3.Cross(rightIndexTip.position - rightPalmPose.position, mainAxis).magnitude;
                
                // Use the larger of the two distances for width
                widthScale = Mathf.Max(leftIndexDistance, rightIndexDistance);
            }
            
            // Ensure minimum width even if detection fails
            widthScale = Mathf.Clamp(widthScale, m_MinWidthScale, m_MaxWidthScale);
            
            // Update static variables with computed values
            HandEllipsoidLength = palmDistance;
            HandEllipsoidWidth = widthScale;
            
            // Show ellipsoid
            ShowEllipsoid(centerPoint, palmDistance, widthScale, mainAxis);
            
            // Debug log to check if we're getting here
            // Debug.Log($"Showing ellipsoid: center={centerPoint}, length={palmDistance}, width={widthScale}");
        }
        
        void ShowEllipsoid(Vector3 position, float length, float width, Vector3 direction)
        {
            // Instead of using cameraOffset, add the SwarmCamera's position.
            GameObject swarmCamera = GameObject.Find("SwarmCamera");
            if (swarmCamera != null)
            {
                position += swarmCamera.transform.position;
            }
            
            // (Rest of the method remains unchanged...)
            if (m_ShapePrefab == null)
            {
                if (m_ShapeInstance == null)
                {
                    Debug.Log("No shape prefab assigned. Creating default sphere.");
                    m_ShapeInstance = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    
                    // Set up a semi-transparent blue material as default
                    Renderer renderer = m_ShapeInstance.GetComponent<Renderer>();
                    if (renderer != null)
                    {
                        Material material = new Material(Shader.Find("Standard"));
                        material.color = new Color(0.3f, 0.5f, 1.0f, 0.5f);
                        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                        material.SetInt("_ZWrite", 0);
                        material.DisableKeyword("_ALPHATEST_ON");
                        material.EnableKeyword("_ALPHABLEND_ON");
                        material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                        material.renderQueue = 3000;
                        renderer.material = material;
                    }
                }
            }
            else if (m_ShapeInstance == null)
            {
                m_ShapeInstance = Instantiate(m_ShapePrefab);
                Debug.Log($"Created new shape instance from prefab: {m_ShapePrefab.name}");
            }
            
            if (m_ShapeInstance != null)
            {
                m_ShapeInstance.transform.position = position;
            
                // Align the rotation based on the specified direction
                Quaternion rotation = Quaternion.FromToRotation(Vector3.forward, direction);
                m_ShapeInstance.transform.rotation = rotation;
            
                // Scale the ellipsoid (X and Y are width, Z is the length)
                m_ShapeInstance.transform.localScale = new Vector3(width * 2, width * 2, length);
            
                if (!m_IsShapeActive)
                {
                    m_ShapeInstance.SetActive(true);
                    m_IsShapeActive = true;
                    Debug.Log("Shape is now active");
                }
            }
        }
        
        void HideShape()
        {
            if (m_ShapeInstance != null && m_IsShapeActive)
            {
                m_ShapeInstance.SetActive(false);
                m_IsShapeActive = false;
            }
            // ADDED: Reset static values when shape is hidden
            HandEllipsoidLength = 0f;
            HandEllipsoidWidth = 0f;
            LeftPalmPosition = Vector3.zero;
            RightPalmPosition = Vector3.zero;
            ArePalmsTracked = false;
        }

        void Update()
        {
            if (m_Subsystem != null)
                return;

            SubsystemManager.GetSubsystems(s_SubsystemsReuse);
            if (s_SubsystemsReuse.Count == 0)
                return;

            m_Subsystem = s_SubsystemsReuse[0];
            m_Subsystem.RegisterProcessor(this);
        }

        void OnDisable()
        {
            if (m_Subsystem != null)
            {
                m_Subsystem.UnregisterProcessor(this);
                m_Subsystem = null;
            }

            // Clean up shape instance
            if (m_ShapeInstance != null)
            {
                Destroy(m_ShapeInstance);
                m_ShapeInstance = null;
                m_IsShapeActive = false;
            }
        }

        XRHandSubsystem m_Subsystem;
        static List<XRHandSubsystem> s_SubsystemsReuse = new List<XRHandSubsystem>();
    }
}

