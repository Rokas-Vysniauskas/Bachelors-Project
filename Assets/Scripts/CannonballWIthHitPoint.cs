using UnityEngine;

public class CannonballWithHitPoint : MonoBehaviour
{
    [Tooltip("Time in seconds before the ball destroys itself automatically")]
    public float lifeTime = 5f;

    [Tooltip("Optimization: Only check for the wall script if the object has this tag.")]
    public string targetTag = "Destructible";

    private Rigidbody rb;
    private Vector3 velocityBeforePhysics;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        Destroy(gameObject, lifeTime);
    }

    void FixedUpdate()
    {
        if (rb != null)
        {
            velocityBeforePhysics = rb.linearVelocity;
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        bool hitSomething = false;
        Vector3 hitPoint = collision.GetContact(0).point;
        GameObject hitObj = collision.gameObject;

        // Flattened using C# Pattern Matching
        if (hitObj.GetComponentInParent<DestructibleWallVoronoi>() is DestructibleWallVoronoi vWall)
        {
            vWall.BreakWall(hitPoint);
            hitSomething = true;
        }
        else if (hitObj.GetComponentInParent<DestructibleWallSlicing>() is DestructibleWallSlicing sWall)
        {
            sWall.BreakWall(hitPoint);
            hitSomething = true;
        }
        else if (hitObj.GetComponentInParent<DestructibleWall>() is DestructibleWall legacyWall)
        {
            legacyWall.BreakWall();
            hitSomething = true;
        }
        else if (hitObj.GetComponentInParent<FEMDestruction>() != null)
        {
            // FEM handles destruction internally; we just need to know we hit it
            hitSomething = true;
        }

        // Shared Logic: Restore velocity if we broke a wall so the ball keeps flying through
        if (hitSomething && rb != null)
        {
            rb.linearVelocity = velocityBeforePhysics;
        }
    }
}