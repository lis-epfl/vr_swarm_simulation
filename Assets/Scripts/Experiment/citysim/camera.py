"""The FPV camera, the pilot's display and the vehicles' turn rates, in one place.

Mirrored on the C# side by Assets/Scripts/Camera/FpvCameraFov.cs (the diagonal and the feed aspect): change the
two together. analyse.py and the bench read the field of view from here or from there; nothing else should
carry its own copy.
"""
import math

# ----------------------------------------------------------------------------------------- the camera
DIAGONAL_FOV_DEG = 82.1          # DJI Mini 3 Pro spec (ScreenSpawn converts it to Camera.fieldOfView)
FEED_ASPECT = 16 / 9             # every feed size the scenes use (1152 x 648, 1280 x 720, 800 x 450)


def fov_deg(aspect=FEED_ASPECT):
    """(horizontal, vertical) field of view in degrees for an image of this aspect (width / height)."""
    v_half = math.atan(math.tan(math.radians(DIAGONAL_FOV_DEG / 2)) / math.sqrt(aspect ** 2 + 1))
    h_half = math.atan(math.tan(v_half) * aspect)
    return math.degrees(2 * h_half), math.degrees(2 * v_half)


HFOV_DEG, VFOV_DEG = fov_deg()   # 74.4, 46.2
HFOV_HALF = math.radians(HFOV_DEG / 2)
VFOV_HALF = math.radians(VFOV_DEG / 2)

CAMERA_FORWARD_M = 0.4           # FPVCameraScript.offset on DroneReduced.prefab: 0.4 m ahead of the drone
PITCH_MIN_DEG, PITCH_MAX_DEG = -90.0, 60.0   # FPVCameraScript.MinPitch / MaxPitch
# SwarmManager.SetGimbalPitchNormalized maps the joystick dial linearly onto [MinPitch, MaxPitch], so a dial
# left at its centre holds the camera 15 deg down, not level.
DIAL_CENTRE_PITCH_DEG = 0.5 * (PITCH_MIN_DEG + PITCH_MAX_DEG)

# ----------------------------------------------------------------------------------------- the target
HAT_HEIGHT_M = 1.8               # hat centre above the ground (walker origin 0.2 m up + analyse.HAT_HEIGHT 1.6)
HAT_SIZE_M = 0.21                # cowboy crown, ~21 cm across (the largest of the three)
R_ID_M = 40.0                    # a-priori recognition range (analyse.py --range): five hats were identified
#                                  whose closest approach was 27-34 m, so the true limit is >= ~35 m

# ----------------------------------------------------------------------------------------- the display
RING_RADIUS_M = 2.0              # ScreenSpawn OUTER_CIRCLE radius
QUAD_W_M, QUAD_H_M = FEED_ASPECT, 1.0   # a feed quad is (width / height, 1, 1) x scale 1
QUAD_HDEG = math.degrees(2 * math.atan(QUAD_W_M / 2 / RING_RADIUS_M))   # 47.9
QUAD_VDEG = math.degrees(2 * math.atan(QUAD_H_M / 2 / RING_RADIUS_M))   # 28.1
DISPLAY_SCALE = QUAD_HDEG / HFOV_DEG   # degrees on the quad per degree in the camera (~0.64): a 74.4 deg feed on
#                                        a 47.9 deg quad. Both interfaces show their feeds on these quads.
HMD_HALF_FOV_DEG = 50.0          # Quest Pro, horizontally, about the head's forward
VIEW_CONE_HALF_DEG = 80.0        # ScreenSpawn.viewConeHalfAngleDeg: a refresh throttle only, never a hide


def recognition_range(theta_min_deg, hat_m=HAT_SIZE_M, display_scale=DISPLAY_SCALE):
    """Camera-to-hat distance (m) at which the hat subtends theta_min_deg on the displayed quad.

    The recognition criterion is angular size on the display: a hat of width s at distance d spans
    s/d rad in the camera and display_scale times that on the quad. Both interfaces share the quad, so the
    criterion is one distance; R_ID_M = 40 m is theta_min ~ 0.19 deg."""
    return hat_m * display_scale / math.radians(theta_min_deg)


def theta_min_for(range_m, hat_m=HAT_SIZE_M, display_scale=DISPLAY_SCALE):
    """Inverse of recognition_range."""
    return math.degrees(hat_m * display_scale / range_m)


# ----------------------------------------------------------------------------------------- turning
DRONE_MAX_YAW_RATE_DEG = 75.0    # VelocityControl.maxYawRate 1.309 rad/s (Mini 3 Pro)
BODY_YAW_RATE_DEG = 90.0         # PyUniSharingFast bodyYawRate: rotates the whole rig (swarm)
HEAD_TURN_RATE_DEG = 180.0       # a comfortable head turn, used by the swarm's reveal model
HEAD_TURN_MAX_DEG = 90.0         # how far a seated pilot turns the head before body yaw takes over
MAX_SPEED_MS = 9.31              # VelocityControl.maxSpeed (DroneReduced.prefab)
CRUISE_SPEED_MS = 7.6            # drag-limited cruise at full stick
