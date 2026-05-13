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

    [Tooltip("Packs the cylindrical rings tighter near the center. 1 = linear, 3 = highly dense center.")]
    [Range(1f, 5f)] public float focus = 2.5f;

    public int cylinderSlices = 4;
    public int radialSlices = 8;

    [Tooltip("Randomness of the plane angles to break 2D symmetry.")]
    [Range(0f, 1f)] public float planeChaos = 0.5f;

    [Tooltip("Number of global 3D cuts that slice through the entire fracture zone. Utterly destroys spider-web shapes.")]
    [Range(0, 3)] public int extraCrossCuts = 2;

    [Header("Physics Settings")]
    public float explosionForce = 600f;
    public float explosionRadius = 4f;

    [Tooltip("Shrinks ONLY the physics collider to prevent the 'Pizza Slice' explosion without creating visual gaps.")]
    [Range(0.8f, 0.99f)] public float collisionShrink = 0.95f;

    public string projectileTag = "CannonBall";

    [Header("Materials")]
    public Material outsideMaterial;
    public Material insideMaterial;
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

        ClipPlane[] spokes = new ClipPlane[radialSlices];
        float angleStep = (Mathf.PI * 2f) / radialSlices;
        for (int r = 0; r < radialSlices; r++)
        {
            float a = r * angleStep + Random.Range(-angleStep * 0.4f, angleStep * 0.4f) * planeChaos;
            Vector3 dir = (tangent * Mathf.Cos(a)) + (bitangent * Mathf.Sin(a));
            Vector3 normal = Vector3.Cross(wallNormal, dir).normalized;

            float tilt = Random.Range(-45f, 45f) * planeChaos;
            normal = Quaternion.AngleAxis(tilt, dir) * normal;

            spokes[r] = new ClipPlane { normal = normal, distance = Vector3.Dot(impact, normal) };
        }

        ClipPlane[,] rings = new ClipPlane[cylinderSlices, radialSlices];
        float[] radii = new float[cylinderSlices];
        for (int c = 1; c < cylinderSlices; c++)
        {
            radii[c] = impactRadius * Mathf.Pow((float)c / cylinderSlices, focus);
        }

        for (int c = 1; c < cylinderSlices; c++)
        {
            for (int r = 0; r < radialSlices; r++)
            {
                float aMid = (r + 0.5f) * angleStep;
                Vector3 dir = (tangent * Mathf.Cos(aMid)) + (bitangent * Mathf.Sin(aMid));
                Vector3 pt = impact + dir * radii[c];

                float tilt = Random.Range(-40f, 40f) * planeChaos;
                Vector3 normal = Quaternion.AngleAxis(tilt, Vector3.Cross(dir, wallNormal)) * dir;

                rings[c, r] = new ClipPlane { normal = normal, distance = Vector3.Dot(pt, normal) };
            }
        }

        List<FractureCell> baseCells = new List<FractureCell>();
        for (int c = 0; c < cylinderSlices; c++)
        {
            for (int r = 0; r < radialSlices; r++)
            {
                FractureCell cell = new FractureCell();
                cell.planeCount = 0;

                SetPlane(ref cell, spokes[r]);

                int nextR = (r + 1) % radialSlices;
                ClipPlane rightSpoke = spokes[nextR];
                SetPlane(ref cell, new ClipPlane { normal = -rightSpoke.normal, distance = -rightSpoke.distance });

                if (c > 0)
                {
                    SetPlane(ref cell, rings[c, r]);
                }

                if (c < cylinderSlices - 1)
                {
                    ClipPlane outerRing = rings[c + 1, r];
                    SetPlane(ref cell, new ClipPlane { normal = -outerRing.normal, distance = -outerRing.distance });
                }

                baseCells.Add(cell);
            }
        }

        List<FractureCell> finalCells = new List<FractureCell>(baseCells);
        for (int i = 0; i < extraCrossCuts; i++)
        {
            Vector3 randNormal = Random.onUnitSphere;
            Vector3 pt = impact + Random.insideUnitSphere * (impactRadius * 0.3f);
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
                else
                {
                    splitCells.Add(cell);
                }
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

            // Separate collision structures
            List<Vector3> colVerts = new List<Vector3>();
            List<int> colTris = new List<int>();
            Dictionary<Vector3, int> colWeldMap = new Dictionary<Vector3, int>();

            foreach (var t in tris)
            {
                Vector3 lv0 = t.v0 - centerOfMass;
                Vector3 lv1 = t.v1 - centerOfMass;
                Vector3 lv2 = t.v2 - centerOfMass;

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

            // =================================================================================
            // THE FIX: "Collision ShrinkWrap"
            // We pull the collision vertices inward by the collisionShrink percentage.
            // This leaves the Visual Mesh perfectly 100% flush, but creates a safe physical 
            // gap so the needle-thin pizza-slice tips don't overlap infinitely at the center.
            // =================================================================================
            for (int i = 0; i < colVerts.Count; i++)
            {
                colVerts[i] = colVerts[i] * collisionShrink;
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

            // Visual scale is strictly 1.0. No ugly gaps will be visible before the explosion.
            chunkObj.transform.localScale = transform.localScale;

            chunkObj.AddComponent<MeshFilter>().mesh = visMesh;
            MeshRenderer mr = chunkObj.AddComponent<MeshRenderer>();
            mr.materials = new Material[] { outsideMaterial, insideMaterial };

            MeshCollider col = chunkObj.AddComponent<MeshCollider>();
            col.convex = true;
            col.sharedMesh = colMesh;
            if (rubblePhysicsMaterial != null) col.material = rubblePhysicsMaterial;

            Rigidbody rb = chunkObj.AddComponent<Rigidbody>();
            float chunkVolume = CalculateVolume(tris);
            rb.mass = Mathf.Max(totalWallMass * (chunkVolume / totalVolume), 0.05f);

            // Backup safety: Keep maximum sliding velocity low just in case geometry gets weird
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
        Vector3 roundedN = new Vector3(
            Mathf.Round(n.x * 100f) / 100f,
            Mathf.Round(n.y * 100f) / 100f,
            Mathf.Round(n.z * 100f) / 100f
        );

        var key = (roundedV, roundedN);
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