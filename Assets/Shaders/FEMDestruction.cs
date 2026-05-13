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
    public float damping = 5f;
    public float yieldStress = 1500f;
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

    // Collision state
    private bool hasImpactThisFrame = false;
    private Vector3 currentImpactPoint;
    private Vector3 currentImpactVelocity;
    private float currentImpactRadius;
    private float currentImpactMass;

    // State trackers
    private bool isReadbackPending = false;
    private bool isWallHidden = false;
    private NativeArray<Node> localNodeData;
    private Bounds localBounds;

    private void Start()
    {
        localBounds = GetComponent<MeshFilter>().sharedMesh.bounds;
        InitializeFEMGrid();
    }

    private void InitializeFEMGrid()
    {
        totalNodes = nodeResolution.x * nodeResolution.y * nodeResolution.z;
        Node[] initialNodes = new Node[totalNodes];

        calculatedNodeSpacing = new Vector3(
            localBounds.size.x / Mathf.Max(1, nodeResolution.x - 1),
            localBounds.size.y / Mathf.Max(1, nodeResolution.y - 1),
            localBounds.size.z / Mathf.Max(1, nodeResolution.z - 1)
        );

        Vector3 originOffset = localBounds.min;

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
        // Safe exit using fake-null check to prevent the Editor from throwing errors if destroyed
        if (this == null || request.hasError)
        {
            isReadbackPending = false;
            return;
        }

        // Must check active state separately after confirming 'this' is not null
        if (!gameObject.activeInHierarchy)
        {
            isReadbackPending = false;
            return;
        }

        request.GetData<Node>().CopyTo(localNodeData);
        bool bufferNeedsUpdate = false;

        for (int i = 0; i < totalNodes; i++)
        {
            Node node = localNodeData[i];

            if (node.isBroken == 1)
            {
                // The moment the first node breaks, hide the original wall and turn off its collider
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
            }
        }

        if (bufferNeedsUpdate)
        {
            nodeBuffer.SetData(localNodeData);
        }

        isReadbackPending = false;
    }

    private void SpawnDebris(Node node)
    {
        if (debrisPrefab == null)
        {
            Debug.LogWarning("FEM Destruction: No Debris Prefab assigned! Cannot spawn rubble.");
            return;
        }

        GameObject debris = Instantiate(debrisPrefab, node.position, Quaternion.identity);
        debris.transform.localScale = calculatedNodeSpacing;

        Rigidbody rb = debris.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = node.velocity;
        }
    }

    private void OnDestroy()
    {
        if (nodeBuffer != null) nodeBuffer.Release();
        if (localNodeData.IsCreated) localNodeData.Dispose();
    }
}