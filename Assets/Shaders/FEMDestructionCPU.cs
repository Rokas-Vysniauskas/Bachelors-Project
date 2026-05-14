using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using System.Runtime.InteropServices;

[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(Collider))]
public class FEMDestructionCPU : MonoBehaviour
{
    [Header("Compute Shader (REQUIRED)")]
    public ComputeShader femCompute;

    [Header("Destruction Visuals")]
    public GameObject debrisPrefab;
    [Range(0, 5)] public int meshSmoothingIterations = 2;
    [Range(0f, 1f)] public float vertexJitterAmount = 0.4f;

    [Header("FEM Resolution")]
    public Vector3Int nodeResolution = new Vector3Int(40, 40, 10);

    [Header("Structural Integrity")]
    [Tooltip("Bolts the bottom row of nodes to the ground so the wall has a foundation.")]
    public bool anchorBottomToGround = true;
    [Tooltip("Bolts the sides of the nodes to the environment.")]
    public bool anchorSidesToWalls = false;
    [Tooltip("How far a chunk can be pushed from its original spot before snapping into debris.")]
    public float maxDeformationDistance = 0.6f;

    [Header("Physics Settings")]
    public float stiffness = 5000f;
    public float damping = 15f;
    public float yieldStress = 500f;
    public float impactForceMultiplier = 1.0f;

    [Header("Physics Filter")]
    public string projectileTag = "Cannonball";

    [StructLayout(LayoutKind.Sequential)]
    private struct Node
    {
        public Vector3 position;
        public Vector3 restPosition;
        public Vector3 velocity;
        public float stress;
        public int isBroken;
        public int isAnchor; // Updated to match HLSL
    }

    private ComputeBuffer nodeBuffer;
    private int totalNodes;
    private int kernelApplyImpact, kernelCalculateStress, kernelIntegrate;
    private Vector3Int threadGroups;
    private Vector3 calculatedNodeSpacing;

    private bool hasImpactThisFrame = false;
    private Vector3 currentImpactPoint, currentImpactVelocity;
    private float currentImpactRadius, currentImpactMass;

    private bool isReadbackPending = false;
    private bool isWallHidden = false;
    private NativeArray<Node> localNodeData;
    private Bounds localBounds;

    private GameObject proceduralWallObj;
    private Mesh proceduralMesh;

    private void Start()
    {
        localBounds = GetComponent<MeshFilter>().sharedMesh.bounds;
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null) rb.isKinematic = true;

        InitializeFEMGrid();
    }

    private void InitializeFEMGrid()
    {
        totalNodes = nodeResolution.x * nodeResolution.y * nodeResolution.z;
        Node[] initialNodes = new Node[totalNodes];

        calculatedNodeSpacing = new Vector3(
            localBounds.size.x / Mathf.Max(1, nodeResolution.x),
            localBounds.size.y / Mathf.Max(1, nodeResolution.y),
            localBounds.size.z / Mathf.Max(1, nodeResolution.z)
        );

        Vector3 originOffset = localBounds.min + (calculatedNodeSpacing / 2f);

        for (int z = 0; z < nodeResolution.z; z++)
        {
            for (int y = 0; y < nodeResolution.y; y++)
            {
                for (int x = 0; x < nodeResolution.x; x++)
                {
                    int index = x + y * nodeResolution.x + z * nodeResolution.x * nodeResolution.y;
                    Vector3 localPos = originOffset + new Vector3(x * calculatedNodeSpacing.x, y * calculatedNodeSpacing.y, z * calculatedNodeSpacing.z);
                    Vector3 worldPos = transform.TransformPoint(localPos);

                    int isAnchored = 0;
                    if (anchorBottomToGround && y == 0) isAnchored = 1;
                    if (anchorSidesToWalls && (x == 0 || x == nodeResolution.x - 1)) isAnchored = 1;

                    initialNodes[index] = new Node
                    {
                        position = worldPos,
                        restPosition = worldPos,
                        velocity = Vector3.zero,
                        stress = 0f,
                        isBroken = 0,
                        isAnchor = isAnchored
                    };
                }
            }
        }

        int stride = Marshal.SizeOf(typeof(Node));
        nodeBuffer = new ComputeBuffer(totalNodes, stride);
        nodeBuffer.SetData(initialNodes);

        kernelApplyImpact = femCompute.FindKernel("ApplyImpact");
        kernelCalculateStress = femCompute.FindKernel("CalculateStress");
        kernelIntegrate = femCompute.FindKernel("IntegrateAndFracture");

        femCompute.SetBuffer(kernelApplyImpact, "_Nodes", nodeBuffer);
        femCompute.SetBuffer(kernelCalculateStress, "_Nodes", nodeBuffer);
        femCompute.SetBuffer(kernelIntegrate, "_Nodes", nodeBuffer);

        femCompute.SetInts("_GridResolution", nodeResolution.x, nodeResolution.y, nodeResolution.z);
        femCompute.SetVector("_NodeSpacing", calculatedNodeSpacing);

        threadGroups = new Vector3Int(
            Mathf.CeilToInt(nodeResolution.x / 8f),
            Mathf.CeilToInt(nodeResolution.y / 8f),
            Mathf.CeilToInt(nodeResolution.z / 8f)
        );

        localNodeData = new NativeArray<Node>(totalNodes, Allocator.Persistent);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (!string.IsNullOrEmpty(projectileTag) && !collision.gameObject.CompareTag(projectileTag)) return;

        Rigidbody rb = collision.rigidbody;
        if (rb != null && collision.contacts.Length > 0)
        {
            hasImpactThisFrame = true;
            currentImpactPoint = collision.contacts[0].point;
            currentImpactVelocity = rb.linearVelocity * impactForceMultiplier;
            currentImpactMass = rb.mass;
            currentImpactRadius = collision.collider != null ? collision.collider.bounds.extents.magnitude : 1.0f;
        }
    }

    private void FixedUpdate()
    {
        if (nodeBuffer == null) return;
        DispatchComputeShader();
    }

    private void Update()
    {
        if (nodeBuffer == null) return;
        RequestAsyncReadback();
    }

    private void DispatchComputeShader()
    {
        femCompute.SetFloat("_DeltaTime", Time.fixedDeltaTime);
        femCompute.SetFloat("_Stiffness", stiffness);
        femCompute.SetFloat("_Damping", damping);
        femCompute.SetFloat("_YieldStress", yieldStress);
        femCompute.SetFloat("_MaxDisplacement", maxDeformationDistance);

        if (hasImpactThisFrame)
        {
            femCompute.SetInt("_HasImpact", 1);
            femCompute.SetVector("_ImpactPoint", currentImpactPoint);
            femCompute.SetVector("_ImpactVelocity", currentImpactVelocity);
            femCompute.SetFloat("_ImpactRadius", currentImpactRadius);
            femCompute.SetFloat("_ImpactMass", currentImpactMass);
            femCompute.Dispatch(kernelApplyImpact, threadGroups.x, threadGroups.y, threadGroups.z);
            hasImpactThisFrame = false;
        }
        else femCompute.SetInt("_HasImpact", 0);

        femCompute.Dispatch(kernelCalculateStress, threadGroups.x, threadGroups.y, threadGroups.z);
        femCompute.Dispatch(kernelIntegrate, threadGroups.x, threadGroups.y, threadGroups.z);
    }

    private void RequestAsyncReadback()
    {
        if (!isReadbackPending)
        {
            isReadbackPending = true;
            AsyncGPUReadback.Request(nodeBuffer, OnReadbackComplete);
        }
    }

    private void OnReadbackComplete(AsyncGPUReadbackRequest request)
    {
        if (this == null || !gameObject.activeInHierarchy || request.hasError)
        {
            isReadbackPending = false;
            return;
        }

        request.GetData<Node>().CopyTo(localNodeData);
        bool bufferNeedsUpdate = false;
        bool needsMeshRebuild = false;

        for (int i = 0; i < totalNodes; i++)
        {
            Node node = localNodeData[i];
            if (node.isBroken == 1)
            {
                if (!isWallHidden)
                {
                    GetComponent<MeshRenderer>().enabled = false;
                    GetComponent<Collider>().enabled = false;
                    isWallHidden = true;
                }

                SpawnDebris(node);

                Node updatedNode = node;
                updatedNode.isBroken = 2;
                localNodeData[i] = updatedNode;

                bufferNeedsUpdate = true;
                needsMeshRebuild = true;
            }
        }

        if (needsMeshRebuild) RebuildProceduralMesh();
        if (bufferNeedsUpdate) nodeBuffer.SetData(localNodeData);

        isReadbackPending = false;
    }

    private void RebuildProceduralMesh()
    {
        if (proceduralMesh == null)
        {
            proceduralMesh = new Mesh();
            proceduralMesh.indexFormat = IndexFormat.UInt32;

            if (proceduralWallObj == null)
            {
                proceduralWallObj = new GameObject("FEM_UnbrokenWall");
                proceduralWallObj.transform.SetParent(transform);
                proceduralWallObj.transform.localPosition = Vector3.zero;
                proceduralWallObj.transform.localRotation = Quaternion.identity;
                proceduralWallObj.transform.localScale = Vector3.one;

                var mf = proceduralWallObj.AddComponent<MeshFilter>();
                mf.mesh = proceduralMesh;
                var mr = proceduralWallObj.AddComponent<MeshRenderer>();
                mr.materials = GetComponent<MeshRenderer>().sharedMaterials;

                var col = proceduralWallObj.AddComponent<MeshCollider>();
                col.convex = true;
            }
        }

        // OPTIMIZATION: Pre-allocating list sizes prevents C# from constantly resizing memory arrays
        List<Vector3> verts = new List<Vector3>(totalNodes / 4);
        List<int> tris = new List<int>(totalNodes / 2);

        Vector3 h = calculatedNodeSpacing / 2f;
        Vector3 originOffset = localBounds.min + h;

        for (int z = 0; z < nodeResolution.z; z++)
        {
            for (int y = 0; y < nodeResolution.y; y++)
            {
                for (int x = 0; x < nodeResolution.x; x++)
                {
                    int i = x + y * nodeResolution.x + z * nodeResolution.x * nodeResolution.y;
                    if (localNodeData[i].isBroken >= 1) continue;

                    Vector3 center = originOffset + new Vector3(x * calculatedNodeSpacing.x, y * calculatedNodeSpacing.y, z * calculatedNodeSpacing.z);

                    if (IsEmpty(x + 1, y, z)) AddQuad(center, h, 5, 1, 2, 6, verts, tris);
                    if (IsEmpty(x - 1, y, z)) AddQuad(center, h, 0, 4, 7, 3, verts, tris);
                    if (IsEmpty(x, y + 1, z)) AddQuad(center, h, 3, 7, 6, 2, verts, tris);
                    if (IsEmpty(x, y - 1, z)) AddQuad(center, h, 0, 1, 5, 4, verts, tris);
                    if (IsEmpty(x, y, z + 1)) AddQuad(center, h, 4, 5, 6, 7, verts, tris);
                    if (IsEmpty(x, y, z - 1)) AddQuad(center, h, 1, 0, 3, 2, verts, tris);
                }
            }
        }

        proceduralMesh.Clear();
        proceduralMesh.SetVertices(verts);
        proceduralMesh.SetTriangles(tris, 0);

        if (vertexJitterAmount > 0f) ApplyVertexJitter(proceduralMesh);
        if (meshSmoothingIterations > 0) SmoothCraterMesh(proceduralMesh, meshSmoothingIterations);

        proceduralMesh.RecalculateNormals();

        var colMesh = proceduralWallObj.GetComponent<MeshCollider>();
        colMesh.sharedMesh = null;
        colMesh.sharedMesh = proceduralMesh;
    }

    private void ApplyVertexJitter(Mesh mesh)
    {
        Vector3[] vertices = mesh.vertices;
        float maxJitter = calculatedNodeSpacing.magnitude * vertexJitterAmount;

        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 pos = vertices[i];

            if (Mathf.Abs(pos.x - localBounds.max.x) < 0.05f || Mathf.Abs(pos.x - localBounds.min.x) < 0.05f ||
                Mathf.Abs(pos.y - localBounds.max.y) < 0.05f || Mathf.Abs(pos.y - localBounds.min.y) < 0.05f ||
                Mathf.Abs(pos.z - localBounds.max.z) < 0.05f || Mathf.Abs(pos.z - localBounds.min.z) < 0.05f)
            {
                continue;
            }

            float noiseX = (Mathf.PerlinNoise(pos.y * 15f, pos.z * 15f) - 0.5f) * 2f;
            float noiseY = (Mathf.PerlinNoise(pos.x * 15f, pos.z * 15f) - 0.5f) * 2f;
            float noiseZ = (Mathf.PerlinNoise(pos.x * 15f, pos.y * 15f) - 0.5f) * 2f;

            vertices[i] += new Vector3(noiseX, noiseY, noiseZ) * maxJitter;
        }

        mesh.vertices = vertices;
    }

    private void SmoothCraterMesh(Mesh mesh, int iterations)
    {
        Vector3[] vertices = mesh.vertices;
        Dictionary<Vector3, List<int>> sharedVertices = new Dictionary<Vector3, List<int>>();

        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 rounded = new Vector3(Mathf.Round(vertices[i].x * 1000f) / 1000f, Mathf.Round(vertices[i].y * 1000f) / 1000f, Mathf.Round(vertices[i].z * 1000f) / 1000f);
            if (!sharedVertices.ContainsKey(rounded)) sharedVertices[rounded] = new List<int>();
            sharedVertices[rounded].Add(i);
        }

        for (int iter = 0; iter < iterations; iter++)
        {
            Vector3[] newVertices = (Vector3[])vertices.Clone();
            foreach (var kvp in sharedVertices)
            {
                Vector3 pos = kvp.Key;
                if (Mathf.Abs(pos.x - localBounds.max.x) < 0.1f || Mathf.Abs(pos.x - localBounds.min.x) < 0.1f ||
                    Mathf.Abs(pos.y - localBounds.max.y) < 0.1f || Mathf.Abs(pos.y - localBounds.min.y) < 0.1f ||
                    Mathf.Abs(pos.z - localBounds.max.z) < 0.1f || Mathf.Abs(pos.z - localBounds.min.z) < 0.1f)
                {
                    continue;
                }

                Vector3 sum = Vector3.zero;
                foreach (int index in kvp.Value) sum += vertices[index];
                Vector3 average = sum / kvp.Value.Count;

                foreach (int index in kvp.Value) newVertices[index] = Vector3.Lerp(vertices[index], average, 0.5f);
            }
            vertices = newVertices;
        }
        mesh.vertices = vertices;
    }

    private bool IsEmpty(int x, int y, int z)
    {
        if (x < 0 || x >= nodeResolution.x || y < 0 || y >= nodeResolution.y || z < 0 || z >= nodeResolution.z) return true;
        int idx = x + y * nodeResolution.x + z * nodeResolution.x * nodeResolution.y;
        return localNodeData[idx].isBroken >= 1;
    }

    private void AddQuad(Vector3 c, Vector3 h, int i0, int i1, int i2, int i3, List<Vector3> verts, List<int> tris)
    {
        Vector3[] p = new Vector3[8];
        p[0] = c + new Vector3(-h.x, -h.y, -h.z);
        p[1] = c + new Vector3(h.x, -h.y, -h.z);
        p[2] = c + new Vector3(h.x, h.y, -h.z);
        p[3] = c + new Vector3(-h.x, h.y, -h.z);
        p[4] = c + new Vector3(-h.x, -h.y, h.z);
        p[5] = c + new Vector3(h.x, -h.y, h.z);
        p[6] = c + new Vector3(h.x, h.y, h.z);
        p[7] = c + new Vector3(-h.x, h.y, h.z);

        int idx = verts.Count;
        verts.Add(p[i0]); verts.Add(p[i1]); verts.Add(p[i2]); verts.Add(p[i3]);
        tris.Add(idx); tris.Add(idx + 1); tris.Add(idx + 2);
        tris.Add(idx); tris.Add(idx + 2); tris.Add(idx + 3);
    }

    private void SpawnDebris(Node node)
    {
        if (debrisPrefab == null) return;

        Quaternion randomRot = Quaternion.Euler(Random.Range(0, 360), Random.Range(0, 360), Random.Range(0, 360));
        GameObject debris = Instantiate(debrisPrefab, node.position, randomRot);

        float randomScale = Random.Range(0.8f, 1.4f);
        debris.transform.localScale = calculatedNodeSpacing * randomScale;

        Rigidbody rb = debris.GetComponent<Rigidbody>();
        if (rb != null) rb.linearVelocity = node.velocity;
    }

    private void OnDisable()
    {
        if (nodeBuffer != null)
        {
            nodeBuffer.Release();
            nodeBuffer = null;
        }
        if (localNodeData.IsCreated) localNodeData.Dispose();
    }
}