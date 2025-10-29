using UnityEngine;

/// <summary>
/// Utility script to diagnose and fix visibility issues with objects that have colliders.
/// This script helps ensure objects are properly visible to cameras while maintaining collision detection.
/// </summary>
public class VisibilityDiagnostic : MonoBehaviour
{
    [Header("Diagnostic Settings")]
    public bool autoFixVisibility = true;
    public bool showDebugInfo = true;
    public Material defaultMaterial; // Assign a default material for invisible objects
    
    [Header("Layer Settings")]
    public int obstacleLayer = 10; // Layer for obstacle detection
    public bool setToObstacleLayer = true;
    
    [ContextMenu("Diagnose Object")]
    public void DiagnoseObject()
    {
        GameObject obj = this.gameObject;
        
        Debug.Log($"=== Diagnosing {obj.name} ===");
        
        // Check components
        Collider col = obj.GetComponent<Collider>();
        MeshRenderer meshRenderer = obj.GetComponent<MeshRenderer>();
        MeshFilter meshFilter = obj.GetComponent<MeshFilter>();
        
        Debug.Log($"Has Collider: {col != null}");
        Debug.Log($"Has MeshRenderer: {meshRenderer != null}");
        Debug.Log($"Has MeshFilter: {meshFilter != null}");
        Debug.Log($"Current Layer: {obj.layer} ({LayerMask.LayerToName(obj.layer)})");
        
        if (col != null)
        {
            Debug.Log($"Collider Type: {col.GetType().Name}");
            Debug.Log($"Collider Enabled: {col.enabled}");
            Debug.Log($"Is Trigger: {col.isTrigger}");
        }
        
        if (meshRenderer != null)
        {
            Debug.Log($"Renderer Enabled: {meshRenderer.enabled}");
            Debug.Log($"Materials Count: {meshRenderer.materials.Length}");
            Debug.Log($"Shadows: {meshRenderer.shadowCastingMode}");
        }
        
        // Auto-fix if enabled
        if (autoFixVisibility)
        {
            FixVisibility();
        }
    }
    
    [ContextMenu("Fix Visibility")]
    public void FixVisibility()
    {
        GameObject obj = this.gameObject;
        bool changes = false;
        
        Debug.Log($"Fixing visibility for {obj.name}...");
        
        // 1. Set correct layer if needed
        if (setToObstacleLayer && obj.layer != obstacleLayer)
        {
            obj.layer = obstacleLayer;
            Debug.Log($"Set layer to {obstacleLayer} (Obstacle layer)");
            changes = true;
        }
        
        // 2. Ensure MeshRenderer exists
        MeshRenderer meshRenderer = obj.GetComponent<MeshRenderer>();
        if (meshRenderer == null)
        {
            meshRenderer = obj.AddComponent<MeshRenderer>();
            Debug.Log("Added MeshRenderer component");
            changes = true;
        }
        
        // 3. Ensure MeshFilter exists
        MeshFilter meshFilter = obj.GetComponent<MeshFilter>();
        if (meshFilter == null)
        {
            meshFilter = obj.AddComponent<MeshFilter>();
            Debug.Log("Added MeshFilter component");
            changes = true;
        }
        
        // 4. Check if we have a mesh
        if (meshFilter.mesh == null)
        {
            // Try to create a mesh from the collider
            Collider col = obj.GetComponent<Collider>();
            if (col != null)
            {
                CreateMeshFromCollider(col, meshFilter);
                changes = true;
            }
        }
        
        // 5. Ensure we have a material
        if (meshRenderer.material == null || meshRenderer.material.name.Contains("Default"))
        {
            if (defaultMaterial != null)
            {
                meshRenderer.material = defaultMaterial;
                Debug.Log("Applied default material");
                changes = true;
            }
            else
            {
                // Create a simple colored material
                Material simpleMaterial = CreateSimpleMaterial();
                meshRenderer.material = simpleMaterial;
                Debug.Log("Created and applied simple material");
                changes = true;
            }
        }
        
        // 6. Ensure renderer is enabled
        if (!meshRenderer.enabled)
        {
            meshRenderer.enabled = true;
            Debug.Log("Enabled MeshRenderer");
            changes = true;
        }
        
        if (changes)
        {
            Debug.Log($"✅ Fixed visibility issues for {obj.name}");
        }
        else
        {
            Debug.Log($"✅ No visibility issues found for {obj.name}");
        }
    }
    
    void CreateMeshFromCollider(Collider col, MeshFilter meshFilter)
    {
        if (col is BoxCollider boxCol)
        {
            // Create a cube mesh for box collider
            meshFilter.mesh = CreateCubeMesh(boxCol.size);
            Debug.Log("Created cube mesh from BoxCollider");
        }
        else if (col is SphereCollider sphereCol)
        {
            // Create a sphere mesh for sphere collider
            meshFilter.mesh = CreateSphereMesh(sphereCol.radius);
            Debug.Log("Created sphere mesh from SphereCollider");
        }
        else if (col is MeshCollider meshCol && meshCol.sharedMesh != null)
        {
            // Use the mesh from mesh collider
            meshFilter.mesh = meshCol.sharedMesh;
            Debug.Log("Used mesh from MeshCollider");
        }
        else
        {
            // Default to a unit cube
            meshFilter.mesh = CreateCubeMesh(Vector3.one);
            Debug.Log("Created default cube mesh");
        }
    }
    
    Mesh CreateCubeMesh(Vector3 size)
    {
        Mesh mesh = new Mesh();
        mesh.name = "Generated Cube";
        
        // Create vertices for a cube
        Vector3[] vertices = new Vector3[8];
        Vector3 halfSize = size * 0.5f;
        
        vertices[0] = new Vector3(-halfSize.x, -halfSize.y, -halfSize.z);
        vertices[1] = new Vector3(halfSize.x, -halfSize.y, -halfSize.z);
        vertices[2] = new Vector3(halfSize.x, halfSize.y, -halfSize.z);
        vertices[3] = new Vector3(-halfSize.x, halfSize.y, -halfSize.z);
        vertices[4] = new Vector3(-halfSize.x, -halfSize.y, halfSize.z);
        vertices[5] = new Vector3(halfSize.x, -halfSize.y, halfSize.z);
        vertices[6] = new Vector3(halfSize.x, halfSize.y, halfSize.z);
        vertices[7] = new Vector3(-halfSize.x, halfSize.y, halfSize.z);
        
        // Create triangles
        int[] triangles = new int[36] {
            // Front face
            0, 2, 1, 0, 3, 2,
            // Back face
            4, 5, 6, 4, 6, 7,
            // Left face
            0, 4, 7, 0, 7, 3,
            // Right face
            1, 6, 5, 1, 2, 6,
            // Top face
            3, 7, 6, 3, 6, 2,
            // Bottom face
            0, 1, 5, 0, 5, 4
        };
        
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        
        return mesh;
    }
    
    Mesh CreateSphereMesh(float radius)
    {
        // For simplicity, create a primitive sphere and return its mesh
        GameObject tempSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Mesh sphereMesh = tempSphere.GetComponent<MeshFilter>().mesh;
        
        // Scale vertices to match radius
        Vector3[] vertices = sphereMesh.vertices;
        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i] *= radius;
        }
        sphereMesh.vertices = vertices;
        
        DestroyImmediate(tempSphere);
        return sphereMesh;
    }
    
    Material CreateSimpleMaterial()
    {
        Material material = new Material(Shader.Find("Standard"));
        material.color = new Color(0.8f, 0.6f, 0.4f, 1f); // Light brown color
        material.name = "Generated House Material";
        return material;
    }
    
    [ContextMenu("Check Camera Visibility")]
    public void CheckCameraVisibility()
    {
        // Find all cameras and check if this object is visible to them
        Camera[] cameras = FindObjectsOfType<Camera>();
        
        Debug.Log($"=== Camera Visibility Check for {gameObject.name} ===");
        
        foreach (Camera cam in cameras)
        {
            bool isVisible = IsVisibleToCamera(cam);
            Debug.Log($"Camera '{cam.name}': {(isVisible ? "✅ Visible" : "❌ Not Visible")}");
            
            if (!isVisible)
            {
                // Check specific reasons
                CheckVisibilityReasons(cam);
            }
        }
    }
    
    bool IsVisibleToCamera(Camera camera)
    {
        Renderer renderer = GetComponent<Renderer>();
        if (renderer == null) return false;
        
        // Check if layer is in camera's culling mask
        int layerMask = 1 << gameObject.layer;
        if ((camera.cullingMask & layerMask) == 0)
            return false;
        
        // Check if within camera's view frustum
        Bounds bounds = renderer.bounds;
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(camera);
        return GeometryUtility.TestPlanesAABB(planes, bounds);
    }
    
    void CheckVisibilityReasons(Camera camera)
    {
        Debug.Log($"  Checking visibility issues for camera '{camera.name}':");
        
        // Check layer mask
        int layerMask = 1 << gameObject.layer;
        bool inCullingMask = (camera.cullingMask & layerMask) != 0;
        Debug.Log($"    Layer in culling mask: {inCullingMask}");
        
        // Check renderer
        Renderer renderer = GetComponent<Renderer>();
        Debug.Log($"    Has renderer: {renderer != null}");
        if (renderer != null)
        {
            Debug.Log($"    Renderer enabled: {renderer.enabled}");
        }
        
        // Check distance
        float distance = Vector3.Distance(camera.transform.position, transform.position);
        Debug.Log($"    Distance to camera: {distance:F2} units");
        Debug.Log($"    Camera far plane: {camera.farClipPlane}");
    }
}