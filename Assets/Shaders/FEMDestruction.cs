using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using System.Runtime.InteropServices;

[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(Collider))]
public class FEMDestruction : MonoBehaviour
{
    [Header("Compute Shader (REQUIRED)")]
    public ComputeShader femCompute;

    [Header("Destruction Visuals")]
    public GameObject debrisPrefab;
    [Range(0f, 1f)] public float vertexJitterAmount = 0.4f;

    [Header("FEM Resolution")]
    public Vector3Int nodeResolution = new Vector3Int(40, 40, 10);

    [Header("Structural Integrity")]
    public bool anchorBottomToGround = true;
    public bool anchorSidesToWalls = false;
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
        public Vector3 position, restPosition, velocity;
        public float stress;
        public int isBroken, isAnchor;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct OutputTriangle
    {
        public Vector3 v0, v1, v2, normal;
        public int isValid;
    }

    private ComputeBuffer nodeBuffer;
    private ComputeBuffer triangleBuffer;
    private int totalNodes;
    private int maxTriangles;
    private int kernelApplyImpact, kernelCalculateStress, kernelIntegrate, kernelGenerateMesh, kernelClearMesh;
    private Vector3Int threadGroups;
    private Vector3 calculatedNodeSpacing;

    private bool hasImpactThisFrame = false;
    private Vector3 currentImpactPoint, currentImpactVelocity;
    private float currentImpactRadius, currentImpactMass;

    private bool isPhysicsReadbackPending = false;
    private bool isMeshReadbackPending = false;
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

        int nodeStride = Marshal.SizeOf(typeof(Node));
        nodeBuffer = new ComputeBuffer(totalNodes, nodeStride);
        nodeBuffer.SetData(initialNodes);

        maxTriangles = totalNodes * 12;
        int triStride = Marshal.SizeOf(typeof(OutputTriangle));
        triangleBuffer = new ComputeBuffer(maxTriangles, triStride, ComputeBufferType.Append);

        kernelApplyImpact = femCompute.FindKernel("ApplyImpact");
        kernelCalculateStress = femCompute.FindKernel("CalculateStress");
        kernelIntegrate = femCompute.FindKernel("IntegrateAndFracture");
        kernelGenerateMesh = femCompute.FindKernel("GenerateWallMesh");
        kernelClearMesh = femCompute.FindKernel("ClearWallMesh");

        femCompute.SetBuffer(kernelApplyImpact, "_Nodes", nodeBuffer);
        femCompute.SetBuffer(kernelCalculateStress, "_Nodes", nodeBuffer);
        femCompute.SetBuffer(kernelIntegrate, "_Nodes", nodeBuffer);

        femCompute.SetBuffer(kernelGenerateMesh, "_Nodes", nodeBuffer);
        femCompute.SetBuffer(kernelGenerateMesh, "_WallTriangles", triangleBuffer);

        femCompute.SetInts("_GridResolution", nodeResolution.x, nodeResolution.y, nodeResolution.z);
        femCompute.SetVector("_NodeSpacing", calculatedNodeSpacing);
        femCompute.SetVector("_BoxMin", localBounds.min);
        femCompute.SetVector("_BoxMax", localBounds.max);

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
        RequestPhysicsReadback();
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

    private void RequestPhysicsReadback()
    {
        if (!isPhysicsReadbackPending)
        {
            isPhysicsReadbackPending = true;
            AsyncGPUReadback.Request(nodeBuffer, OnPhysicsReadbackComplete);
        }
    }

    private void OnPhysicsReadbackComplete(AsyncGPUReadbackRequest request)
    {
        if (this == null || !gameObject.activeInHierarchy || request.hasError)
        {
            isPhysicsReadbackPending = false;
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

        if (bufferNeedsUpdate) nodeBuffer.SetData(localNodeData);
        isPhysicsReadbackPending = false;

        if (needsMeshRebuild && !isMeshReadbackPending)
        {
            RequestGPUWallMesh();
        }
    }

    private void RequestGPUWallMesh()
    {
        isMeshReadbackPending = true;

        // 1. Instantly wipe the buffer clean on the GPU (No ghost triangles!)
        femCompute.SetBuffer(kernelClearMesh, "_WallTrianglesRW", triangleBuffer);
        femCompute.SetInt("_MaxTriangles", maxTriangles);
        int threadGroupsClear = Mathf.CeilToInt(maxTriangles / 64f);
        femCompute.Dispatch(kernelClearMesh, threadGroupsClear, 1, 1);

        // 2. Reset Append Counter
        triangleBuffer.SetCounterValue(0);

        // 3. Generate fresh mesh
        femCompute.SetFloat("_JitterAmount", vertexJitterAmount);
        femCompute.Dispatch(kernelGenerateMesh, threadGroups.x, threadGroups.y, threadGroups.z);

        AsyncGPUReadback.Request(triangleBuffer, OnMeshReadbackComplete);
    }

    private void OnMeshReadbackComplete(AsyncGPUReadbackRequest request)
    {
        if (this == null || !gameObject.activeInHierarchy || request.hasError)
        {
            isMeshReadbackPending = false;
            return;
        }

        OutputTriangle[] gpuTriangles = request.GetData<OutputTriangle>().ToArray();

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

        List<Vector3> verts = new List<Vector3>(gpuTriangles.Length * 3);
        List<int> tris = new List<int>(gpuTriangles.Length * 3);

        for (int i = 0; i < gpuTriangles.Length; i++)
        {
            // The ultimate validity check. Rejects any uninitialized memory.
            if (gpuTriangles[i].isValid == 0) continue;

            int idx = verts.Count;
            verts.Add(gpuTriangles[i].v0);
            verts.Add(gpuTriangles[i].v1);
            verts.Add(gpuTriangles[i].v2);

            tris.Add(idx);
            tris.Add(idx + 1);
            tris.Add(idx + 2);
        }

        proceduralMesh.Clear();
        proceduralMesh.SetVertices(verts);
        proceduralMesh.SetTriangles(tris, 0);
        proceduralMesh.RecalculateNormals();

        var colMesh = proceduralWallObj.GetComponent<MeshCollider>();
        colMesh.sharedMesh = null;
        colMesh.sharedMesh = proceduralMesh;

        isMeshReadbackPending = false;
    }

    private void SpawnDebris(Node node)
    {
        if (debrisPrefab == null) return;

        // Safety Catch: Prevent any rogue NaN physics glitches from crashing the Unity Editor
        if (float.IsNaN(node.position.x) || float.IsNaN(node.velocity.x)) return;

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
        if (triangleBuffer != null)
        {
            triangleBuffer.Release();
            triangleBuffer = null;
        }
        if (localNodeData.IsCreated) localNodeData.Dispose();
    }
}