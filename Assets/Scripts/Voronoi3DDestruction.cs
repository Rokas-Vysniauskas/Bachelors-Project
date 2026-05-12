using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
public class Voronoi3DDestruction : MonoBehaviour
{
    [Header("Destruction Settings")]
    [Tooltip("Number of true 3D volumetric pieces.")]
    public int pieceCount = 40;

    [Tooltip("How tightly the shards cluster at the impact point.")]
    [Range(0f, 1f)] public float impactBias = 0.8f;

    public float explosionForce = 600f;
    public float explosionRadius = 4f;

    [Header("Materials")]
    [Tooltip("Material for the original outside faces of the wall.")]
    public Material outsideMaterial;
    [Tooltip("Material for the new inside cut faces of the concrete/stone.")]
    public Material insideMaterial;

    [Header("Physics Filter")]
    public string projectileTag = "CannonBall";

    private bool isBroken = false;
    private float totalWallMass = 100f;

    // --- Helper Classes for 3D Polygon Math ---
    private class Polygon
    {
        public List<Vector3> verts;
        public Vector3 normal;
        public bool isSurface; // True if this face was part of the original wall exterior
    }

    private class Polyhedron
    {
        public List<Polygon> faces = new List<Polygon>();
    }

    void Start()
    {
        if (outsideMaterial == null) outsideMaterial = GetComponent<MeshRenderer>().sharedMaterial;
        if (insideMaterial == null) insideMaterial = outsideMaterial;

        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null) totalWallMass = rb.mass;
    }

    void OnCollisionEnter(Collision collision)
    {
        if (isBroken) return;
        if (!string.IsNullOrEmpty(projectileTag) && !collision.gameObject.CompareTag(projectileTag)) return;

        if (collision.contacts.Length > 0)
        {
            isBroken = true;
            GetComponent<Collider>().enabled = false;
            GetComponent<MeshRenderer>().enabled = false;

            GenerateTrue3DChunks(collision.contacts[0].point);
        }
    }

    private void GenerateTrue3DChunks(Vector3 worldImpactPoint)
    {
        Bounds bounds = GetComponent<MeshRenderer>().bounds; // World space bounds

        // 1. Generate 3D Voronoi Seeds clustered around impact
        List<Vector3> seeds = new List<Vector3>();
        for (int i = 0; i < pieceCount; i++)
        {
            Vector3 randomPoint = new Vector3(
                Random.Range(bounds.min.x, bounds.max.x),
                Random.Range(bounds.min.y, bounds.max.y),
                Random.Range(bounds.min.z, bounds.max.z)
            );
            Vector3 biasedPoint = Vector3.Lerp(randomPoint, worldImpactPoint, Random.Range(0f, impactBias));
            seeds.Add(biasedPoint);
        }

        // 2. Define the Base Wall as a Polyhedron
        Polyhedron baseWall = CreateBoundingBox(bounds);
        float totalCalculatedVolume = 0f;
        List<Polyhedron> finalChunks = new List<Polyhedron>();

        // 3. Slice the Box for every seed (True 3D Voronoi algorithm)
        for (int i = 0; i < seeds.Count; i++)
        {
            Polyhedron cell = baseWall;
            Vector3 currentSeed = seeds[i];

            for (int j = 0; j < seeds.Count; j++)
            {
                if (i == j) continue;

                Vector3 otherSeed = seeds[j];
                Vector3 planeNormal = (otherSeed - currentSeed).normalized;
                Vector3 planePoint = (currentSeed + otherSeed) * 0.5f;

                cell = ClipPolyhedron(cell, planePoint, planeNormal);
                if (cell.faces.Count < 4) break; // Degenerate sliver
            }

            if (cell.faces.Count >= 4)
            {
                finalChunks.Add(cell);
                totalCalculatedVolume += CalculateVolume(cell);
            }
        }

        // Prevent divide by zero error if math fails
        if (totalCalculatedVolume <= 0.001f) totalCalculatedVolume = 1f;

        // 4. Build the GameObjects
        foreach (var chunk in finalChunks)
        {
            BuildChunkMesh(chunk, totalCalculatedVolume, worldImpactPoint);
        }

        Destroy(gameObject, 5f); // Cleanup original wall
    }

    // Mathematically slices a 3D shape in half and caps the hole
    private Polyhedron ClipPolyhedron(Polyhedron input, Vector3 planePt, Vector3 planeN)
    {
        Polyhedron result = new Polyhedron();
        List<Vector3> intersectionEdges = new List<Vector3>();

        foreach (var poly in input.faces)
        {
            List<Vector3> keptVerts = new List<Vector3>();
            for (int i = 0; i < poly.verts.Count; i++)
            {
                Vector3 a = poly.verts[i];
                Vector3 b = poly.verts[(i + 1) % poly.verts.Count];

                bool aIn = Vector3.Dot(a - planePt, planeN) <= 0.0001f;
                bool bIn = Vector3.Dot(b - planePt, planeN) <= 0.0001f;

                if (aIn) keptVerts.Add(a);

                if (aIn != bIn)
                {
                    float t = Vector3.Dot(planePt - a, planeN) / Vector3.Dot(b - a, planeN);
                    Vector3 intersect = Vector3.Lerp(a, b, t);
                    keptVerts.Add(intersect);
                    intersectionEdges.Add(intersect);
                }
            }

            if (keptVerts.Count >= 3)
            {
                result.faces.Add(new Polygon { verts = keptVerts, normal = poly.normal, isSurface = poly.isSurface });
            }
        }

        // Cap the hole created by the slice
        if (intersectionEdges.Count >= 3)
        {
            // Filter unique points
            List<Vector3> uniquePts = new List<Vector3>();
            foreach (var p in intersectionEdges)
            {
                bool isDup = false;
                foreach (var u in uniquePts)
                {
                    if (Vector3.Distance(p, u) < 0.001f) { isDup = true; break; }
                }
                if (!isDup) uniquePts.Add(p);
            }

            if (uniquePts.Count >= 3)
            {
                // Sort the points circularly to form a valid convex face
                Vector3 center = Vector3.zero;
                foreach (var p in uniquePts) center += p;
                center /= uniquePts.Count;

                Vector3 right = (uniquePts[0] - center).normalized;
                Vector3 forward = Vector3.Cross(planeN, right).normalized;

                uniquePts.Sort((a, b) =>
                {
                    float angleA = Mathf.Atan2(Vector3.Dot(a - center, forward), Vector3.Dot(a - center, right));
                    float angleB = Mathf.Atan2(Vector3.Dot(b - center, forward), Vector3.Dot(b - center, right));
                    return angleA.CompareTo(angleB); // Ascending sort = correct winding against plane normal
                });

                // This is a new internal cut face, so isSurface = false
                result.faces.Add(new Polygon { verts = uniquePts, normal = planeN, isSurface = false });
            }
        }
        return result;
    }

    private void BuildChunkMesh(Polyhedron p, float totalVolume, Vector3 impactPos)
    {
        // Calculate Center of Mass (Pivot)
        Vector3 pivot = Vector3.zero;
        int ptCount = 0;
        foreach (var f in p.faces) foreach (var v in f.verts) { pivot += v; ptCount++; }
        pivot /= ptCount;

        List<Vector3> verts = new List<Vector3>();
        List<Vector3> normals = new List<Vector3>();
        List<int> outsideTris = new List<int>();
        List<int> insideTris = new List<int>();

        // Build the Mesh Data
        foreach (var face in p.faces)
        {
            int baseIdx = verts.Count;
            // Shift vertices to local space around the pivot
            foreach (var v in face.verts)
            {
                verts.Add(v - pivot);
                normals.Add(face.normal);
            }

            // Triangulate face
            for (int i = 1; i < face.verts.Count - 1; i++)
            {
                if (face.isSurface)
                {
                    outsideTris.Add(baseIdx); outsideTris.Add(baseIdx + i); outsideTris.Add(baseIdx + i + 1);
                }
                else
                {
                    insideTris.Add(baseIdx); insideTris.Add(baseIdx + i); insideTris.Add(baseIdx + i + 1);
                }
            }
        }

        Mesh mesh = new Mesh();
        mesh.vertices = verts.ToArray();
        mesh.normals = normals.ToArray();
        mesh.subMeshCount = 2;
        mesh.SetTriangles(outsideTris, 0);
        mesh.SetTriangles(insideTris, 1);
        mesh.RecalculateBounds();

        // Instantiate
        GameObject chunkObj = new GameObject("3D_Voronoi_Shard");
        chunkObj.transform.position = pivot; // Spawns perfectly in place!
        chunkObj.transform.rotation = Quaternion.identity;

        chunkObj.AddComponent<MeshFilter>().mesh = mesh;
        MeshRenderer mr = chunkObj.AddComponent<MeshRenderer>();
        mr.materials = new Material[] { outsideMaterial, insideMaterial };

        MeshCollider col = chunkObj.AddComponent<MeshCollider>();
        col.convex = true; // Works perfectly now because the mesh is a solid volume!
        col.sharedMesh = mesh;

        Rigidbody rb = chunkObj.AddComponent<Rigidbody>();

        // Exact mass calculation based on volume percentage
        float chunkMass = totalWallMass * (CalculateVolume(p) / totalVolume);
        rb.mass = Mathf.Max(chunkMass, 0.05f); // Prevent physics bugs on tiny slivers

        rb.AddExplosionForce(explosionForce, impactPos, explosionRadius);
    }

    // Defines the base 6-sided wall to carve out of
    private Polyhedron CreateBoundingBox(Bounds b)
    {
        Polyhedron box = new Polyhedron();
        Vector3 v000 = new Vector3(b.min.x, b.min.y, b.min.z);
        Vector3 v100 = new Vector3(b.max.x, b.min.y, b.min.z);
        Vector3 v010 = new Vector3(b.min.x, b.max.y, b.min.z);
        Vector3 v110 = new Vector3(b.max.x, b.max.y, b.min.z);
        Vector3 v001 = new Vector3(b.min.x, b.min.y, b.max.z);
        Vector3 v101 = new Vector3(b.max.x, b.min.y, b.max.z);
        Vector3 v011 = new Vector3(b.min.x, b.max.y, b.max.z);
        Vector3 v111 = new Vector3(b.max.x, b.max.y, b.max.z);

        box.faces.Add(MakePoly(new Vector3[] { v010, v110, v100, v000 }, Vector3.back));    // Front (-Z)
        box.faces.Add(MakePoly(new Vector3[] { v111, v011, v001, v101 }, Vector3.forward)); // Back (+Z)
        box.faces.Add(MakePoly(new Vector3[] { v011, v111, v110, v010 }, Vector3.up));      // Top (+Y)
        box.faces.Add(MakePoly(new Vector3[] { v000, v100, v101, v001 }, Vector3.down));    // Bottom (-Y)
        box.faces.Add(MakePoly(new Vector3[] { v110, v111, v101, v100 }, Vector3.right));   // Right (+X)
        box.faces.Add(MakePoly(new Vector3[] { v011, v010, v000, v001 }, Vector3.left));    // Left (-X)
        return box;
    }

    private Polygon MakePoly(Vector3[] pts, Vector3 normal)
    {
        return new Polygon { verts = new List<Vector3>(pts), normal = normal, isSurface = true };
    }

    // Sums the volume of tetrahedrons drawn from the centroid to each face
    private float CalculateVolume(Polyhedron p)
    {
        float vol = 0;
        Vector3 d = Vector3.zero;
        int ptCount = 0;
        foreach (var f in p.faces) foreach (var v in f.verts) { d += v; ptCount++; }
        d /= ptCount;

        foreach (var f in p.faces)
        {
            for (int i = 1; i < f.verts.Count - 1; i++)
            {
                Vector3 a = f.verts[0], b = f.verts[i], c = f.verts[i + 1];
                vol += Mathf.Abs(Vector3.Dot(a - d, Vector3.Cross(b - d, c - d))) / 6.0f;
            }
        }
        return vol;
    }
}