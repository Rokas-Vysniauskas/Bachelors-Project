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
    [Tooltip("Prefab representing a fractured piece of the wall. REQUIRED! A simple Cube prefab works best.")]
    public GameObject debrisPrefab;

    [Header("FEM Resolution")]
    [Tooltip("How many nodes to generate along the X, Y, and Z axis of the mesh bounds.")]
    public Vector3Int nodeResolution = new Vector3Int(20, 10, 4);

    [Header("Physics Settings")]
    public float stiffness = 5000f;
    public float damping = 15f; // Increased slightly for stability
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
    }

    private ComputeBuffer nodeBuffer;
    private int totalNodes;
    private int kernelApplyImpact;
    private int kernelCalculateStress;
    private int kernelIntegrate;
    private Vector3Int threadGroups;
    private Vector3 calculatedNodeSpacing;

    private bool hasImpactThisFrame = false;
    private Vector3 currentImpactPoint;
    private Vector3 currentImpactVelocity;
    private float currentImpactRadius;
    private float currentImpactMass;

    private bool isReadbackPending = false;
    private bool isWallHidden = false;
    private NativeArray<Node> localNodeData;
    private Bounds localBounds;

    // Procedural Mesh Carving
    private GameObject proceduralWallObj;
    private Mesh proceduralMesh;

    private void Start()
    {
        localBounds = GetComponent<MeshFilter>().sharedMesh.bounds;
        InitializeFEMGrid();
    }

    private void InitializeFEMGrid()
    {
        totalNodes = nodeResolution.x * nodeResolution.y * nodeResolution.z;
        Node[] initialNodes = new Node[totalNodes];

        // Fix: Exact component division perfectly scales the voxels inside the mesh bounds
        calculatedNodeSpacing = new Vector3(
            localBounds.size.x / Mathf.Max(1, nodeResolution.x),
            localBounds.size.y / Mathf.Max(1, nodeResolution.y),
            localBounds.size.z / Mathf.Max(1, nodeResolution.z)
        );

        // Fix: Offset to the center of the first voxel
        Vector3 originOffset = localBounds.min + (calculatedNodeSpacing / 2f);

        for (int z = 0; z < nodeResolution.z; z++)
        {
            for (int y = 0; y < nodeResolution.y; y++)
            {
                for (int x = 0; x < nodeResolution.x; x++)
                {
                    int index = x + y * nodeResolution.x + z * nodeResolution.x * nodeResolution.y;

                    Vector3 localPos = originOffset + new Vector3(
                        x * calculatedNodeSpacing.x,
                        y * calculatedNodeSpacing.y,
                        z * calculatedNodeSpacing.z
                    );

                    Vector3 worldPos = transform.TransformPoint(localPos);

                    initialNodes[index] = new Node
                    {
                        position = worldPos,
                        restPosition = worldPos,
                        velocity = Vector3.zero,
                        stress = 0f,
                        isBroken = 0
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

            Collider col = collision.collider;
            currentImpactRadius = col != null ? col.bounds.extents.magnitude : 1.0f;
        }
    }

    private void Update()
    {
        if (nodeBuffer == null) return;
        DispatchComputeShader();
        RequestAsyncReadback();
    }

    private void DispatchComputeShader()
    {
        femCompute.SetFloat("_DeltaTime", Time.deltaTime);
        femCompute.SetFloat("_Stiffness", stiffness);
        femCompute.SetFloat("_Damping", damping);
        femCompute.SetFloat("_YieldStress", yieldStress);

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
        else
        {
            femCompute.SetInt("_HasImpact", 0);
        }

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

        if (needsMeshRebuild)
        {
            RebuildProceduralMesh();
        }

        if (bufferNeedsUpdate)
        {
            nodeBuffer.SetData(localNodeData);
        }

        isReadbackPending = false;
    }

    // High-Speed Voxel Generator to "scoop out" broken pieces from the original wall bounds
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
                proceduralWallObj.AddComponent<MeshCollider>();
            }
        }

        List<Vector3> verts = new List<Vector3>();
        List<int> tris = new List<int>();
        List<Vector3> norms = new List<Vector3>();
        List<Vector2> uvs = new List<Vector2>();

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

                    Vector3 center = originOffset + new Vector3(
                        x * calculatedNodeSpacing.x,
                        y * calculatedNodeSpacing.y,
                        z * calculatedNodeSpacing.z
                    );

                    // Only draw exterior faces or faces touching empty/broken space
                    if (IsEmpty(x + 1, y, z)) AddQuad(center, h, 5, 1, 2, 6, Vector3.right, verts, tris, norms, uvs);
                    if (IsEmpty(x - 1, y, z)) AddQuad(center, h, 0, 4, 7, 3, Vector3.left, verts, tris, norms, uvs);
                    if (IsEmpty(x, y + 1, z)) AddQuad(center, h, 3, 7, 6, 2, Vector3.up, verts, tris, norms, uvs);
                    if (IsEmpty(x, y - 1, z)) AddQuad(center, h, 0, 1, 5, 4, Vector3.down, verts, tris, norms, uvs);
                    if (IsEmpty(x, y, z + 1)) AddQuad(center, h, 4, 5, 6, 7, Vector3.forward, verts, tris, norms, uvs);
                    if (IsEmpty(x, y, z - 1)) AddQuad(center, h, 1, 0, 3, 2, Vector3.back, verts, tris, norms, uvs);
                }
            }
        }

        proceduralMesh.Clear();
        proceduralMesh.SetVertices(verts);
        proceduralMesh.SetTriangles(tris, 0);
        proceduralMesh.SetNormals(norms);
        proceduralMesh.SetUVs(0, uvs);

        var col = proceduralWallObj.GetComponent<MeshCollider>();
        col.sharedMesh = null;
        col.sharedMesh = proceduralMesh;
    }

    private bool IsEmpty(int x, int y, int z)
    {
        if (x < 0 || x >= nodeResolution.x || y < 0 || y >= nodeResolution.y || z < 0 || z >= nodeResolution.z)
            return true;

        int idx = x + y * nodeResolution.x + z * nodeResolution.x * nodeResolution.y;
        return localNodeData[idx].isBroken >= 1;
    }

    private void AddQuad(Vector3 c, Vector3 h, int i0, int i1, int i2, int i3, Vector3 norm, List<Vector3> verts, List<int> tris, List<Vector3> norms, List<Vector2> uvs)
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
        norms.Add(norm); norms.Add(norm); norms.Add(norm); norms.Add(norm);

        uvs.Add(new Vector2(0, 0)); uvs.Add(new Vector2(1, 0));
        uvs.Add(new Vector2(1, 1)); uvs.Add(new Vector2(0, 1));

        tris.Add(idx); tris.Add(idx + 1); tris.Add(idx + 2);
        tris.Add(idx); tris.Add(idx + 2); tris.Add(idx + 3);
    }

    private void SpawnDebris(Node node)
    {
        if (debrisPrefab == null) return;

        GameObject debris = Instantiate(debrisPrefab, node.position, Quaternion.identity);
        debris.transform.localScale = calculatedNodeSpacing;

        Rigidbody rb = debris.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = node.velocity;
        }
    }

    // Fix: Moved to OnDisable to ensure the leak is patched even if destroyed unexpectedly 
    private void OnDisable()
    {
        if (nodeBuffer != null)
        {
            nodeBuffer.Release();
            nodeBuffer = null;
        }
        if (localNodeData.IsCreated)
        {
            localNodeData.Dispose();
        }
    }
}