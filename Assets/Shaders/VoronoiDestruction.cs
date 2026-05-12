using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(Collider))]
public class VoronoiDestruction : MonoBehaviour
{
    [Header("Compute Shader (REQUIRED)")]
    public ComputeShader voronoiCompute;

    [Header("Destruction Settings")]
    public int pieceCount = 40;
    [Range(0f, 1f)] public float impactBias = 0.8f;
    public float explosionForce = 600f;
    public float explosionRadius = 4f;

    [Header("Materials")]
    public Material outsideMaterial;
    public Material insideMaterial;

    [Header("Physics Filter")]
    public string projectileTag = "CannonBall";

    private bool isBroken = false;
    private float totalWallMass = 100f;
    private Bounds localBounds;

    [StructLayout(LayoutKind.Sequential)]
    public struct OutputTriangle
    {
        public Vector3 v0;
        public Vector3 v1;
        public Vector3 v2;
        public Vector3 normal;
        public int isExterior;
        public int cellID;
    }

    private const int STRIDE = 56;

    void Start()
    {
        if (outsideMaterial == null) outsideMaterial = GetComponent<MeshRenderer>().sharedMaterial;
        if (insideMaterial == null) insideMaterial = outsideMaterial;

        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null) totalWallMass = rb.mass;

        localBounds = GetComponent<MeshFilter>().sharedMesh.bounds;
    }

    void OnCollisionEnter(Collision collision)
    {
        if (isBroken) return;
        if (!string.IsNullOrEmpty(projectileTag) && !collision.gameObject.CompareTag(projectileTag)) return;

        if (voronoiCompute == null)
        {
            Debug.LogError("CRITICAL ERROR: Voronoi Compute Shader is missing!", this);
            return;
        }

        if (collision.contacts.Length > 0)
        {
            isBroken = true;
            GetComponent<Collider>().enabled = false;
            ExecuteGPUFractureAsync(collision.contacts[0].point);
        }
    }

    private void ExecuteGPUFractureAsync(Vector3 worldImpactPoint)
    {
        Vector3 localImpactPoint = transform.InverseTransformPoint(worldImpactPoint);

        Vector3[] seeds = new Vector3[pieceCount];
        for (int i = 0; i < pieceCount; i++)
        {
            Vector3 randomPoint = new Vector3(
                Random.Range(localBounds.min.x, localBounds.max.x),
                Random.Range(localBounds.min.y, localBounds.max.y),
                Random.Range(localBounds.min.z, localBounds.max.z)
            );
            seeds[i] = Vector3.Lerp(randomPoint, localImpactPoint, Random.Range(0f, impactBias));
        }

        // 24 faces * 22 triangles per face (since 24 verts = 22 triangles)
        int maxTriangles = pieceCount * 24 * 22;

        ComputeBuffer seedBuffer = new ComputeBuffer(pieceCount, sizeof(float) * 3);
        seedBuffer.SetData(seeds);

        ComputeBuffer triangleBuffer = new ComputeBuffer(maxTriangles, STRIDE, ComputeBufferType.Append);

        OutputTriangle[] emptyData = new OutputTriangle[maxTriangles];
        for (int i = 0; i < maxTriangles; i++) emptyData[i].cellID = -1;
        triangleBuffer.SetData(emptyData);
        triangleBuffer.SetCounterValue(0);

        int kernel = voronoiCompute.FindKernel("GenerateVoronoiGeometry");
        voronoiCompute.SetBuffer(kernel, "_Seeds", seedBuffer);
        voronoiCompute.SetBuffer(kernel, "_OutputTriangles", triangleBuffer);
        voronoiCompute.SetInt("_SeedCount", pieceCount);
        voronoiCompute.SetVector("_BoxMin", localBounds.min);
        voronoiCompute.SetVector("_BoxMax", localBounds.max);

        // Divide by 8f instead of 64f to match the new [numthreads(8, 1, 1)]
        int threadGroups = Mathf.CeilToInt(pieceCount / 8f);
        voronoiCompute.Dispatch(kernel, threadGroups, 1, 1);

        UnityEngine.Rendering.AsyncGPUReadback.Request(triangleBuffer, request =>
        {
            if (this == null) return;
            OnReadbackComplete(request, worldImpactPoint, seedBuffer, triangleBuffer);
        });
    }

    private void OnReadbackComplete(UnityEngine.Rendering.AsyncGPUReadbackRequest request, Vector3 worldImpactPoint, ComputeBuffer seedBuffer, ComputeBuffer triangleBuffer)
    {
        seedBuffer.Release();
        triangleBuffer.Release();

        if (!gameObject.activeInHierarchy || request.hasError) return;

        OutputTriangle[] gpuTriangles = request.GetData<OutputTriangle>().ToArray();
        GetComponent<MeshRenderer>().enabled = false;

        BuildDualMeshChunks(gpuTriangles, worldImpactPoint);
    }

    private void BuildDualMeshChunks(OutputTriangle[] gpuTriangles, Vector3 worldImpactPoint)
    {
        Dictionary<int, List<OutputTriangle>> chunks = new Dictionary<int, List<OutputTriangle>>();
        foreach (var tri in gpuTriangles)
        {
            if (tri.cellID == -1) continue;
            if (!chunks.ContainsKey(tri.cellID)) chunks[tri.cellID] = new List<OutputTriangle>();
            chunks[tri.cellID].Add(tri);
        }

        float totalVolume = 0f;
        foreach (var tris in chunks.Values) totalVolume += CalculateVolume(tris);
        if (totalVolume <= 0.0001f) totalVolume = 1f;

        foreach (var kvp in chunks)
        {
            List<OutputTriangle> tris = kvp.Value;
            if (tris.Count < 4) continue;

            Vector3 centerOfMass = Vector3.zero;
            foreach (var t in tris) centerOfMass += t.v0 + t.v1 + t.v2;
            centerOfMass /= (tris.Count * 3);

            List<Vector3> visVerts = new List<Vector3>();
            List<Vector3> visNorms = new List<Vector3>();
            List<Vector2> visUvs = new List<Vector2>();
            List<int> visOutsideTris = new List<int>();
            List<int> visInsideTris = new List<int>();

            List<Vector3> colVerts = new List<Vector3>();
            List<int> colTris = new List<int>();
            Dictionary<Vector3, int> colWeldMap = new Dictionary<Vector3, int>();

            foreach (var t in tris)
            {
                Vector3 lv0 = t.v0 - centerOfMass;
                Vector3 lv1 = t.v1 - centerOfMass;
                Vector3 lv2 = t.v2 - centerOfMass;

                // 1. Calculate perfectly flat mathematical normal
                Vector3 cross = Vector3.Cross(lv1 - lv0, lv2 - lv0);
                if (cross.sqrMagnitude < 0.000001f) continue;
                Vector3 flatNormal = cross.normalized;

                // 2. Convex Hull Failsafe: Ensure normal ALWAYS faces away from center
                Vector3 faceCenter = (lv0 + lv1 + lv2) / 3f;
                bool isFlipped = Vector3.Dot(flatNormal, faceCenter) < 0;

                if (isFlipped)
                {
                    flatNormal = -flatNormal;
                }

                int vBase = visVerts.Count;
                visVerts.Add(lv0); visVerts.Add(lv1); visVerts.Add(lv2);
                visNorms.Add(flatNormal); visNorms.Add(flatNormal); visNorms.Add(flatNormal);
                visUvs.Add(new Vector2(lv0.x, lv0.y)); visUvs.Add(new Vector2(lv1.x, lv1.y)); visUvs.Add(new Vector2(lv2.x, lv2.y));

                if (t.isExterior == 1)
                {
                    if (isFlipped) visOutsideTris.AddRange(new int[] { vBase, vBase + 2, vBase + 1 });
                    else visOutsideTris.AddRange(new int[] { vBase, vBase + 1, vBase + 2 });
                }
                else
                {
                    if (isFlipped) visInsideTris.AddRange(new int[] { vBase, vBase + 2, vBase + 1 });
                    else visInsideTris.AddRange(new int[] { vBase, vBase + 1, vBase + 2 });
                }

                int c0 = GetOrAddColVert(lv0, colVerts, colWeldMap);
                int c1 = GetOrAddColVert(lv1, colVerts, colWeldMap);
                int c2 = GetOrAddColVert(lv2, colVerts, colWeldMap);

                if (c0 != c1 && c1 != c2 && c0 != c2)
                {
                    if (isFlipped) colTris.AddRange(new int[] { c0, c2, c1 });
                    else colTris.AddRange(new int[] { c0, c1, c2 });
                }
            }

            Mesh visMesh = new Mesh();
            visMesh.vertices = visVerts.ToArray();
            visMesh.normals = visNorms.ToArray();
            visMesh.uv = visUvs.ToArray();
            visMesh.subMeshCount = 2;
            visMesh.SetTriangles(visOutsideTris, 0);
            visMesh.SetTriangles(visInsideTris, 1);
            visMesh.RecalculateBounds();

            Mesh colMesh = new Mesh();
            colMesh.vertices = colVerts.ToArray();
            colMesh.triangles = colTris.ToArray();

            GameObject chunkObj = new GameObject($"Shard_{kvp.Key}");
            chunkObj.transform.position = transform.TransformPoint(centerOfMass);
            chunkObj.transform.rotation = transform.rotation;
            chunkObj.transform.localScale = transform.localScale;

            chunkObj.AddComponent<MeshFilter>().mesh = visMesh;
            MeshRenderer mr = chunkObj.AddComponent<MeshRenderer>();
            mr.materials = new Material[] { outsideMaterial, insideMaterial };

            MeshCollider col = chunkObj.AddComponent<MeshCollider>();
            col.convex = true;
            col.sharedMesh = colMesh;

            Rigidbody rb = chunkObj.AddComponent<Rigidbody>();
            float chunkVolume = CalculateVolume(tris);
            rb.mass = Mathf.Max(totalWallMass * (chunkVolume / totalVolume), 0.05f);

            rb.AddExplosionForce(explosionForce, worldImpactPoint, explosionRadius);
        }

        Destroy(gameObject, 2f);
    }

    private int GetOrAddColVert(Vector3 v, List<Vector3> verts, Dictionary<Vector3, int> map)
    {
        Vector3 rounded = new Vector3(
            Mathf.Round(v.x * 100f) / 100f,
            Mathf.Round(v.y * 100f) / 100f,
            Mathf.Round(v.z * 100f) / 100f
        );
        if (map.TryGetValue(rounded, out int idx)) return idx;

        idx = verts.Count;
        verts.Add(v);
        map[rounded] = idx;
        return idx;
    }

    private float CalculateVolume(List<OutputTriangle> tris)
    {
        float vol = 0;
        foreach (var t in tris)
        {
            vol += Vector3.Dot(t.v0, Vector3.Cross(t.v1, t.v2)) / 6.0f;
        }
        return Mathf.Abs(vol);
    }
}