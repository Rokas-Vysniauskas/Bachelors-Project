using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(Collider))]
public class PlanarDestruction : MonoBehaviour
{
    [Header("Compute Shader (REQUIRED)")]
    public ComputeShader planarCompute;

    [Header("Slicing Geometry Settings")]
    public Vector3 wallNormal = Vector3.forward;
    public float impactRadius = 4f;

    [Tooltip("Packs the cylindrical rings tighter near the center.")]
    [Range(1f, 3f)] public float focus = 1.5f;

    public int cylinderSlices = 5;

    [Tooltip("Target radial slices for the OUTERMOST ring. The inner rings will scale down to a chunky core.")]
    public int radialSlices = 40;

    [Tooltip("Number of slices in the absolute center. Keep this at 3 or 4 to create a stable 'potato' core and prevent pizza-slice physics explosions.")]
    public int coreSlices = 4;

    [Tooltip("Randomizes angles and tilts to completely destroy symmetrical spider-web patterns.")]
    [Range(0f, 1f)] public float planeChaos = 0.6f;

    [Tooltip("Global 3D cuts that slice through the entire fracture zone to create wedge-shaped rubble.")]
    [Range(0, 3)] public int extraCrossCuts = 2;

    [Header("Physics Settings")]
    public float explosionForce = 150f;
    public float explosionRadius = 4f;

    [Tooltip("Prevents razor-thin shards from weighing too little and flying away like feathers from the explosion force.")]
    public float minShardMass = 2.0f;
    public string projectileTag = "CannonBall";

    [Header("Materials")]
    public Material outsideMaterial;
    public Material insideMaterial;
    [Tooltip("Apply a Zero-Bounce, High-Friction Physic Material here for realistic dead-weight concrete.")]
    public PhysicsMaterial rubblePhysicsMaterial;

    private bool isBroken = false;
    private float totalWallMass = 100f;
    private Bounds localBounds;

    [StructLayout(LayoutKind.Sequential)]
    public struct OutputTriangle
    {
        public Vector3 v0, v1, v2;
        public Vector3 normal;
        public int isExterior;
        public int cellID;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ClipPlane
    {
        public Vector3 normal;
        public float distance;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct FractureCell
    {
        public ClipPlane p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
        public int planeCount;
        public Vector3 pad;
    }

    private const int STRIDE = 56;
    private const int CELL_STRIDE = (12 * 16) + 4 + 12;

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
        if (planarCompute == null) return;

        if (collision.contacts.Length > 0)
        {
            isBroken = true;
            GetComponent<Collider>().enabled = false;
            ExecuteGPUFractureAsync(collision.contacts[0].point);
        }
    }

    private void ExecuteGPUFractureAsync(Vector3 worldImpactPoint)
    {
        Vector3 localImpact = transform.InverseTransformPoint(worldImpactPoint);

        FractureCell[] cells = GenerateFracturePlanes(localImpact);
        int cellCount = cells.Length;
        int maxTriangles = cellCount * 20 * 14;

        ComputeBuffer cellBuffer = new ComputeBuffer(cellCount, CELL_STRIDE);
        cellBuffer.SetData(cells);

        ComputeBuffer triangleBuffer = new ComputeBuffer(maxTriangles, STRIDE, ComputeBufferType.Append);
        OutputTriangle[] emptyData = new OutputTriangle[maxTriangles];
        for (int i = 0; i < maxTriangles; i++) emptyData[i].cellID = -1;
        triangleBuffer.SetData(emptyData);
        triangleBuffer.SetCounterValue(0);

        int kernel = planarCompute.FindKernel("GeneratePlanarGeometry");
        planarCompute.SetBuffer(kernel, "_Cells", cellBuffer);
        planarCompute.SetBuffer(kernel, "_OutputTriangles", triangleBuffer);
        planarCompute.SetInt("_CellCount", cellCount);
        planarCompute.SetVector("_BoxMin", localBounds.min);
        planarCompute.SetVector("_BoxMax", localBounds.max);

        planarCompute.Dispatch(kernel, cellCount, 1, 1);

        AsyncGPUReadback.Request(triangleBuffer, request =>
        {
            if (this == null) return;
            OnReadbackComplete(request, worldImpactPoint, cellBuffer, triangleBuffer);
        });
    }

    private FractureCell[] GenerateFracturePlanes(Vector3 impact)
    {
        Vector3 tangent = Mathf.Abs(wallNormal.y) < 0.99f ? Vector3.up : Vector3.right;
        tangent = Vector3.Normalize(Vector3.Cross(wallNormal, tangent));
        Vector3 bitangent = Vector3.Cross(wallNormal, tangent);

        float[] radii = new float[cylinderSlices];
        for (int c = 0; c < cylinderSlices; c++)
        {
            radii[c] = impactRadius * Mathf.Pow((float)(c + 1) / cylinderSlices, focus);
        }

        List<FractureCell> baseCells = new List<FractureCell>();

        for (int c = 0; c <= cylinderSlices; c++)
        {
            float rInner = (c == 0) ? 0f : radii[c - 1];
            float rOuter = (c == cylinderSlices) ? 100000f : radii[c];

            int currentRadialSlices = (c == 0) ? coreSlices : Mathf.Max(coreSlices, Mathf.RoundToInt(radialSlices * ((float)c / cylinderSlices)));
            float angleStep = (Mathf.PI * 2f) / currentRadialSlices;

            float ringAngleOffset = Random.Range(0f, Mathf.PI * 2f);

            for (int r = 0; r < currentRadialSlices; r++)
            {
                float aStart = ringAngleOffset + (r * angleStep);
                float aEnd = ringAngleOffset + ((r + 1) * angleStep);

                float jitter = angleStep * 0.35f * planeChaos;
                aStart += Random.Range(-jitter, jitter);
                aEnd += Random.Range(-jitter, jitter);

                FractureCell cell = new FractureCell();
                cell.planeCount = 0;

                if (c > 0)
                {
                    Vector3 dir = (tangent * Mathf.Cos((aStart + aEnd) / 2f)) + (bitangent * Mathf.Sin((aStart + aEnd) / 2f));
                    Vector3 pt = impact + dir * rInner;
                    Vector3 normal = Quaternion.AngleAxis(Random.Range(-30f, 30f) * planeChaos, Vector3.Cross(dir, wallNormal)) * dir;
                    SetPlane(ref cell, new ClipPlane { normal = normal, distance = Vector3.Dot(pt, normal) });
                }

                if (c < cylinderSlices)
                {
                    Vector3 dir = (tangent * Mathf.Cos((aStart + aEnd) / 2f)) + (bitangent * Mathf.Sin((aStart + aEnd) / 2f));
                    Vector3 pt = impact + dir * rOuter;
                    Vector3 normal = Quaternion.AngleAxis(Random.Range(-30f, 30f) * planeChaos, Vector3.Cross(dir, wallNormal)) * -dir;
                    SetPlane(ref cell, new ClipPlane { normal = normal, distance = Vector3.Dot(pt, normal) });
                }

                Vector3 leftDir = (tangent * Mathf.Cos(aStart)) + (bitangent * Mathf.Sin(aStart));
                Vector3 leftNorm = Vector3.Cross(wallNormal, leftDir).normalized;
                leftNorm = Quaternion.AngleAxis(Random.Range(-45f, 45f) * planeChaos, leftDir) * leftNorm;
                SetPlane(ref cell, new ClipPlane { normal = leftNorm, distance = Vector3.Dot(impact, leftNorm) });

                Vector3 rightDir = (tangent * Mathf.Cos(aEnd)) + (bitangent * Mathf.Sin(aEnd));
                Vector3 rightNorm = Vector3.Cross(rightDir, wallNormal).normalized;
                rightNorm = Quaternion.AngleAxis(Random.Range(-45f, 45f) * planeChaos, rightDir) * rightNorm;
                SetPlane(ref cell, new ClipPlane { normal = rightNorm, distance = Vector3.Dot(impact, rightNorm) });

                baseCells.Add(cell);
            }
        }

        List<FractureCell> finalCells = new List<FractureCell>(baseCells);
        for (int i = 0; i < extraCrossCuts; i++)
        {
            Vector3 randNormal = Random.onUnitSphere;
            Vector3 pt = impact + Random.insideUnitSphere * (impactRadius * 0.2f);
            ClipPlane crossPlane = new ClipPlane { normal = randNormal, distance = Vector3.Dot(pt, randNormal) };
            ClipPlane invCrossPlane = new ClipPlane { normal = -randNormal, distance = -Vector3.Dot(pt, randNormal) };

            List<FractureCell> splitCells = new List<FractureCell>();
            foreach (var cell in finalCells)
            {
                if (cell.planeCount < 12)
                {
                    FractureCell posCell = cell;
                    SetPlane(ref posCell, crossPlane);
                    splitCells.Add(posCell);

                    FractureCell negCell = cell;
                    SetPlane(ref negCell, invCrossPlane);
                    splitCells.Add(negCell);
                }
                else splitCells.Add(cell);
            }
            finalCells = splitCells;
        }

        return finalCells.ToArray();
    }

    private void SetPlane(ref FractureCell cell, ClipPlane plane)
    {
        switch (cell.planeCount)
        {
            case 0: cell.p0 = plane; break;
            case 1: cell.p1 = plane; break;
            case 2: cell.p2 = plane; break;
            case 3: cell.p3 = plane; break;
            case 4: cell.p4 = plane; break;
            case 5: cell.p5 = plane; break;
            case 6: cell.p6 = plane; break;
            case 7: cell.p7 = plane; break;
            case 8: cell.p8 = plane; break;
            case 9: cell.p9 = plane; break;
            case 10: cell.p10 = plane; break;
            case 11: cell.p11 = plane; break;
        }
        cell.planeCount++;
    }

    private void OnReadbackComplete(AsyncGPUReadbackRequest request, Vector3 worldImpactPoint, ComputeBuffer cellBuffer, ComputeBuffer triangleBuffer)
    {
        cellBuffer.Release();
        triangleBuffer.Release();

        if (!gameObject.activeInHierarchy || request.hasError) return;

        OutputTriangle[] gpuTriangles = request.GetData<OutputTriangle>().ToArray();
        GetComponent<MeshRenderer>().enabled = false;

        BuildDualMeshChunks(gpuTriangles, worldImpactPoint);
    }

    private void BuildDualMeshChunks(OutputTriangle[] gpuTriangles, Vector3 worldImpactPoint)
    {
        Dictionary<Vector3, Vector3> globalZipper = new Dictionary<Vector3, Vector3>();
        for (int i = 0; i < gpuTriangles.Length; i++)
        {
            if (gpuTriangles[i].cellID == -1) continue;
            gpuTriangles[i].v0 = GlobalWeld(gpuTriangles[i].v0, globalZipper);
            gpuTriangles[i].v1 = GlobalWeld(gpuTriangles[i].v1, globalZipper);
            gpuTriangles[i].v2 = GlobalWeld(gpuTriangles[i].v2, globalZipper);
        }

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

            Dictionary<(Vector3, Vector3), int> visWeldMap = new Dictionary<(Vector3, Vector3), int>();

            List<Vector3> colVerts = new List<Vector3>();
            List<int> colTris = new List<int>();
            Dictionary<Vector3, int> colWeldMap = new Dictionary<Vector3, int>();

            foreach (var t in tris)
            {
                Vector3 lv0 = t.v0 - centerOfMass;
                Vector3 lv1 = t.v1 - centerOfMass;
                Vector3 lv2 = t.v2 - centerOfMass;

                // We trust the GPU's mathematically perfect normals unconditionally.
                Vector3 flatNormal = t.normal;
                Vector3 faceCenter = (lv0 + lv1 + lv2) / 3f;
                bool isFlipped = Vector3.Dot(flatNormal, faceCenter) < 0;
                if (isFlipped) flatNormal = -flatNormal;

                int v0 = GetOrAddVisVert(lv0, flatNormal, new Vector2(lv0.x, lv0.y), visVerts, visNorms, visUvs, visWeldMap);
                int v1 = GetOrAddVisVert(lv1, flatNormal, new Vector2(lv1.x, lv1.y), visVerts, visNorms, visUvs, visWeldMap);
                int v2 = GetOrAddVisVert(lv2, flatNormal, new Vector2(lv2.x, lv2.y), visVerts, visNorms, visUvs, visWeldMap);

                if (v0 != v1 && v1 != v2 && v0 != v2)
                {
                    if (t.isExterior == 1)
                    {
                        if (isFlipped) visOutsideTris.AddRange(new int[] { v0, v2, v1 });
                        else visOutsideTris.AddRange(new int[] { v0, v1, v2 });
                    }
                    else
                    {
                        if (isFlipped) visInsideTris.AddRange(new int[] { v0, v2, v1 });
                        else visInsideTris.AddRange(new int[] { v0, v1, v2 });
                    }
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

            if (colVerts.Count < 4) continue;

            Mesh visMesh = new Mesh();
            visMesh.vertices = visVerts.ToArray();

            // The GPU-generated normals are directly applied. No overwriting occurs.
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

            chunkObj.AddComponent<MeshFilter>().mesh = visMesh;
            MeshRenderer mr = chunkObj.AddComponent<MeshRenderer>();
            mr.materials = new Material[] { outsideMaterial, insideMaterial };

            MeshCollider col = chunkObj.AddComponent<MeshCollider>();
            col.convex = true;
            col.sharedMesh = colMesh;

            // Physics material guarantees dead-weight concrete behavior
            if (rubblePhysicsMaterial != null) col.material = rubblePhysicsMaterial;

            Rigidbody rb = chunkObj.AddComponent<Rigidbody>();
            float chunkVolume = CalculateVolume(tris);
            float calculatedMass = totalWallMass * (chunkVolume / totalVolume);

            rb.mass = Mathf.Max(calculatedMass, minShardMass);
            rb.maxDepenetrationVelocity = 2.0f;

            if (explosionForce > 0f)
            {
                rb.AddExplosionForce(explosionForce, worldImpactPoint, explosionRadius);
            }
        }

        Destroy(gameObject, 2f);
    }

    private Vector3 GlobalWeld(Vector3 v, Dictionary<Vector3, Vector3> map)
    {
        Vector3 rounded = new Vector3(
            Mathf.Round(v.x * 10000f) / 10000f,
            Mathf.Round(v.y * 10000f) / 10000f,
            Mathf.Round(v.z * 10000f) / 10000f
        );
        if (map.TryGetValue(rounded, out Vector3 truePos)) return truePos;
        map[rounded] = v;
        return v;
    }

    private int GetOrAddVisVert(Vector3 v, Vector3 n, Vector2 uv, List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, Dictionary<(Vector3, Vector3), int> map)
    {
        Vector3 roundedV = new Vector3(
            Mathf.Round(v.x * 10000f) / 10000f,
            Mathf.Round(v.y * 10000f) / 10000f,
            Mathf.Round(v.z * 10000f) / 10000f
        );

        // THE FIX: Exact normals are required as keys. Rounding them forced sharp corners 
        // to share smoothed normals, creating the muddy shadow artifacts.
        var key = (roundedV, n);

        if (map.TryGetValue(key, out int idx)) return idx;

        idx = verts.Count;
        verts.Add(v);
        norms.Add(n);
        uvs.Add(uv);
        map[key] = idx;
        return idx;
    }

    private int GetOrAddColVert(Vector3 v, List<Vector3> verts, Dictionary<Vector3, int> map)
    {
        Vector3 rounded = new Vector3(
            Mathf.Round(v.x * 10000f) / 10000f,
            Mathf.Round(v.y * 10000f) / 10000f,
            Mathf.Round(v.z * 10000f) / 10000f
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
        foreach (var t in tris) vol += Vector3.Dot(t.v0, Vector3.Cross(t.v1, t.v2)) / 6.0f;
        return Mathf.Abs(vol);
    }
}