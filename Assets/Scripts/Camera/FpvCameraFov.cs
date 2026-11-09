using UnityEngine;

/// <summary>
/// The drone FPV camera's field of view, in one place. The DJI Mini 3 Pro is specified by its diagonal
/// (82.1 deg); Unity's <c>Camera.fieldOfView</c> is vertical, so <c>ScreenSpawn</c> converts the diagonal to the
/// vertical angle for the feed's aspect (~46.2 deg at 16:9, ~74.4 deg horizontally) when it sets up each camera.
///
/// <para>Everything that reasons about what a drone camera can see takes the angle from here: the headless bench's
/// look-gap read-outs (<c>SwarmBenchRunner</c>, which used to assume 98.5 deg horizontally — the prefab's 66.3 deg
/// vertical before ScreenSpawn overrides it), and on the Python side <c>analyse.py</c> and the visibility simulator,
/// which import the same two numbers from <c>Assets/Scripts/Experiment/citysim/camera.py</c>. Change both together.</para>
/// </summary>
public static class FpvCameraFov
{
    /// <summary>DJI Mini 3 Pro diagonal field of view, degrees.</summary>
    public const float DiagonalDeg = 82.1f;

    /// <summary>The feeds' aspect (1152 x 648 and every other size the scenes use).</summary>
    public const float FeedAspect = 16f / 9f;

    /// <summary>Vertical field of view, degrees, for an image of this aspect (width / height).</summary>
    public static float VerticalDeg(float aspect)
    {
        float diagHalf = DiagonalDeg * 0.5f * Mathf.Deg2Rad;
        return 2f * Mathf.Atan(Mathf.Tan(diagHalf) / Mathf.Sqrt(aspect * aspect + 1f)) * Mathf.Rad2Deg;
    }

    /// <summary>Horizontal field of view, degrees, for an image of this aspect (width / height).</summary>
    public static float HorizontalDeg(float aspect)
    {
        float vertHalf = VerticalDeg(aspect) * 0.5f * Mathf.Deg2Rad;
        return 2f * Mathf.Atan(Mathf.Tan(vertHalf) * aspect) * Mathf.Rad2Deg;
    }
}
