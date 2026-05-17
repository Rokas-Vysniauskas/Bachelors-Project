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
    [Tooltip("The core logic file running on the GPU.")]
    public ComputeShader femCompute;

    [Header("Destruction Visuals")]
    [Tooltip("The physical object spawned when a chunk breaks. Use a jagged rock prefab, not a cube.")]
    public GameObject debrisPrefab;
    [Tooltip("Scrambles the internal voxel grid to look like jagged, broken concrete. 0 is flat cubes, 1 is extreme spikes.")]
    [Range(0f, 1f)] public float vertexJitterAmount = 0.5f;
    [Tooltip("The Anti-Aliasing effect running on the GPU. 0 = Minecraft blocks. 100 = Completely melted, smooth concrete slopes.")]
    [Range(0f, 100f)] public float craterEdgeSmoothing = 60f;

    [Header("Performance")]
    [Tooltip("How many pieces of rubble the CPU is allowed to spawn in a single frame. Prevents massive lag spikes on impact.")]
    public int maxDebrisSpawnsPerFrame = 50;

    [Header("FEM Resolution")]
    [Tooltip("How many nodes to generate. Higher numbers mean smaller debris but demand more CPU/GPU power.")]
    public Vector3Int nodeResolution = new Vector3Int(40, 40, 6);

    [Header("Structural Integrity")]
    [Tooltip("Bolts the bottom row of nodes to the ground so the wall has a foundation.")]
    public bool anchorBottomToGround = true;
    [Tooltip("Bolts the left and right sides of the wall to the environment.")]
    public bool anchorSidesToWalls = false;
    [Tooltip("If the impact physically stretches a chunk this far from its starting point, it snaps into debris.")]
    public float maxDeformationDistance = 0.6f;

    [Header("Physics Settings")]
    [Tooltip("How rigid the springs are. 5000+ is like concrete. 500 is like jello.")]
    public float stiffness = 5000f;
    [Tooltip("Slows down vibrations inside the wall. High damping stops the wall from wobbling visibly.")]
    public float damping = 15f;
    [Tooltip("The amount of accumulated stress required to break a spring. Lower means bigger craters.")]
    public float yieldStress = 500f;
    [Tooltip("Artificially multiplies the kinetic energy of the incoming projectile.")]
    public float impactForceMultiplier = 1.0f;

    [Header("Physics Filter")]
    [Tooltip("Only break the wall if hit by an object with this tag.")]
    public string projectileTag = "Cannonball";

    [StructLayout(LayoutKind.Sequential)]
    private struct Node
    {
        public Vector3 position, restPosition, velocity;
        public float stress;
        public int isBroken, isAnchor;
        public int anchorDistance;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct OutputTriangle
    {
        public Vector3 v0, v1, v2, normal;
        public int isValid;
    }

    private ComputeBuffer nodeBuffer, triangleBuffer;
    private int totalNodes, maxTriangles;
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

    // The Queue that holds chunks waiting to be spawned as physics objects
    private Queue<Node> debrisSpawnQueue = new Queue<Node>();

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
                        isAnchor = isAnchored,
                        anchorDistance = 0
                    };
                }
            }
        }

        nodeBuffer = new ComputeBuffer(totalNodes, Marshal.SizeOf(typeof(Node)));
        nodeBuffer.SetData(initialNodes);

        maxTriangles = totalNodes * 12;
        triangleBuffer = new ComputeBuffer(maxTriangles, Marshal.SizeOf(typeof(OutputTriangle)), ComputeBufferType.Append);

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

        // Time-Sliced Debris Spawning to prevent FPS drops
        int spawnsThisFrame = 0;
        while (debrisSpawnQueue.Count > 0 && spawnsThisFrame < maxDebrisSpawnsPerFrame)
        {
            SpawnDebris(debrisSpawnQueue.Dequeue());
            spawnsThisFrame++;
        }
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
                // Add broken node to the queue instead of instantly spawning
                debrisSpawnQueue.Enqueue(node);

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

        femCompute.SetBuffer(kernelClearMesh, "_WallTrianglesRW", triangleBuffer);
        femCompute.SetInt("_MaxTriangles", maxTriangles);
        femCompute.Dispatch(kernelClearMesh, Mathf.CeilToInt(maxTriangles / 64f), 1, 1);

        triangleBuffer.SetCounterValue(0);

        femCompute.SetFloat("_JitterAmount", vertexJitterAmount);
        femCompute.SetFloat("_EdgeSmoothing", craterEdgeSmoothing / 100f);

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

        NativeArray<OutputTriangle> gpuTriangles = request.GetData<OutputTriangle>();
        int triangleCount = gpuTriangles.Length;

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

                proceduralWallObj.AddComponent<MeshFilter>().mesh = proceduralMesh;
                proceduralWallObj.AddComponent<MeshRenderer>().materials = GetComponent<MeshRenderer>().sharedMaterials;

                var col = proceduralWallObj.AddComponent<MeshCollider>();
                col.convex = false;
            }
        }

        int validTriCount = 0;
        for (int i = 0; i < triangleCount; i++)
        {
            if (gpuTriangles[i].isValid == 1) validTriCount++;
        }

        Vector3[] verts = new Vector3[validTriCount * 3];
        int[] tris = new int[validTriCount * 3];
        int vIdx = 0;

        for (int i = 0; i < triangleCount; i++)
        {
            if (gpuTriangles[i].isValid == 1)
            {
                OutputTriangle tri = gpuTriangles[i];
                verts[vIdx] = tri.v0;
                verts[vIdx + 1] = tri.v1;
                verts[vIdx + 2] = tri.v2;

                tris[vIdx] = vIdx;
                tris[vIdx + 1] = vIdx + 1;
                tris[vIdx + 2] = vIdx + 2;

                vIdx += 3;
            }
        }

        proceduralMesh.Clear();
        proceduralMesh.SetVertices(verts);
        proceduralMesh.SetTriangles(tris, 0);
        proceduralMesh.RecalculateNormals();

        var colMesh = proceduralWallObj.GetComponent<MeshCollider>();
        colMesh.sharedMesh = null;
        colMesh.sharedMesh = proceduralMesh;

        if (!isWallHidden)
        {
            GetComponent<MeshRenderer>().enabled = false;
            GetComponent<Collider>().enabled = false;
            isWallHidden = true;
        }

        isMeshReadbackPending = false;
    }

    private void SpawnDebris(Node node)
    {
        if (debrisPrefab == null) return;
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
        if (nodeBuffer != null) nodeBuffer.Release();
        if (triangleBuffer != null) triangleBuffer.Release();
        if (localNodeData.IsCreated) localNodeData.Dispose();
    }
}