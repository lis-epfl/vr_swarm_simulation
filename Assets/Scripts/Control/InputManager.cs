using System.ComponentModel;
using System.Collections.Generic;
using UnityEngine;

public class InputManager : MonoBehaviour
{
    public enum InputMode
    {
        KEYBOARD,
        JOYSTICK,
        ANY
    }
    [Header("Input Mode")]
    [SerializeField]
    private InputMode inputMode = InputMode.KEYBOARD;

    public enum CommandFrame
    {
        Body,  // Velocity stick is relative to each drone's heading (yaw).
        World, // Velocity stick maps to fixed world axes regardless of heading.
        VR,    // Velocity stick is relative to the pilot's body heading (the OVRCameraRig yaw,
               // integrated from controller yaw in PyUniSharingFast).
    }
    [Header("Command Frame")]
    [Tooltip("Body: the velocity command moves the drone relative to its own heading. " +
             "World: the velocity command moves the drone along fixed world axes.")]
    [SerializeField]
    private CommandFrame commandFrame = CommandFrame.Body;

    [Header("Keyboard Spread (A/D)")]
    [Tooltip("Hold D to widen the swarm and A to tighten it. Sets the Olfati-Saber d_ref. " +
             "Stays inactive (-1) until first pressed, then holds the last value.")]
    [SerializeField] private float spreadInitial = 7.0f;   // fallback start value if the SwarmManager isn't available
    [SerializeField] private float spreadMin     = 1.0f;
    [SerializeField] private float spreadMax     = 20.0f;
    [SerializeField] private float spreadRate    = 0.5f;   // units per second while a key is held
    private float keyboardSpread = -1.0f;                  // -1 ⇒ no override yet

    // The gimbal dial (s2) as last received; NaN until a packet has arrived, and again once the stream has gone
    // stale. The dial commands the gimbal only when it moves from this value, so connecting the controller (or a
    // restarted bridge, whose knob starts over at 0) never moves the cameras by itself.
    private float gimbalDial = float.NaN;

    // Frame the user velocity command is expressed in. Read by SwarmAlgorithm each tick.
    public CommandFrame ActiveCommandFrame => commandFrame;

    // Runtime helpers so the frame can be flipped from a key bind, UI button, etc.
    public void SetCommandFrame(CommandFrame frame) => commandFrame = frame;
    public void ToggleCommandFrame() =>
        commandFrame = commandFrame == CommandFrame.Body ? CommandFrame.World : CommandFrame.Body;

    private Dictionary<string, float> inputStatus = new Dictionary<string, float>()
    {
        {"throttle", 0.0f},
        {"yaw", 0.0f},
        {"pitch", 0.0f},
        {"roll", 0.0f},
        {"spread", -1.0f}, // Default spread value **Should be improved to be set based on the current algorithm**
        {"userSwitch", -1.0f},
    };

    private Dictionary<string, float> inputStatuRaw = new Dictionary<string, float>()
    {
        {"throttle", 0.0f},
        {"yaw", 0.0f},
        {"pitch", 0.0f},
        {"roll", 0.0f},
        {"spread", -1.0f}, // Default spread value **Should be improved to be set based on the current algorithm**
        {"userSwitch", -1.0f},
    };    
    public static InputManager Instance { get; private set; }
    public Dictionary<string, float> InputStatus => inputStatus; // Expose inputStatus as a read-only property
    public Dictionary<string, float> InputStatusRaw => inputStatuRaw; // Expose inputStatus as a read-only property
    private bool isControlLocked = false; // Flag to lock/unlock control input

    void Awake()
    {
        if (Instance == null)
            Instance = this;
        else
            Destroy(gameObject);
    }

    void Update()
    {
        if (isControlLocked)
        {
            // Set all inputs to zero when control is locked
            inputStatus["throttle"] = 0.0f;
            inputStatus["yaw"] = 0.0f;
            inputStatus["pitch"] = 0.0f;
            inputStatus["roll"] = 0.0f;
            inputStatus["spread"] = -1.0f; // Default spread value
            keyboardSpread = -1.0f;        // forget the integrated spread while locked
            inputStatus["userSwitch"] = -1;
            inputStatuRaw = new Dictionary<string, float>(inputStatus); // Keep raw status in sync
            return;
        }
        if (inputMode == InputMode.KEYBOARD || inputMode == InputMode.ANY)
        {
            inputStatus["throttle"]   = Input.GetAxisRaw("Throttle");
            inputStatus["yaw"]        = Input.GetAxisRaw("Yaw");
            inputStatus["pitch"]      = Input.GetAxisRaw("Pitch");
            inputStatus["roll"]       = Input.GetAxisRaw("Roll");
            inputStatus["userSwitch"] = Input.GetKey(KeyCode.Space) ? 1 : -1;

            // A/D integrate a spread target (Olfati-Saber d_ref). Stays at -1 (no override)
            // until first touched, then holds the last value after the keys are released.
            float spreadDir = (Input.GetKey(KeyCode.D) ? 1f : 0f) - (Input.GetKey(KeyCode.A) ? 1f : 0f);
            if (spreadDir != 0f)
            {
                if (keyboardSpread < 0f)
                    keyboardSpread = SwarmManager.Instance != null
                        ? SwarmManager.Instance.GetDRef()
                        : spreadInitial;
                keyboardSpread = Mathf.Clamp(keyboardSpread + spreadDir * spreadRate * Time.deltaTime,
                                             spreadMin, spreadMax);
            }
            inputStatus["spread"] = keyboardSpread;

            inputStatuRaw = new Dictionary<string, float>(inputStatus);
        }
        if (inputMode == InputMode.JOYSTICK || (inputMode == InputMode.ANY && !Input.anyKeyDown))
        {
            JoystickData joystickData = UDPReceiverManager.sharedJoystickData;
            // The sender stopped (RC dropped, bridge stopped). In ANY mode that is no joystick at all,
            // so the keyboard keeps control instead of being overwritten.
            bool stale = UDPReceiverManager.IsStale;
            if (stale)
                gimbalDial = float.NaN;
            if (joystickData != null && !(stale && inputMode == InputMode.ANY))
            {
                // Stale in JOYSTICK mode: neutral sticks, so the swarm holds instead of flying the last
                // command. Spread, switch and gimbal keep their last values: they are knob positions,
                // and resetting them would snap d_ref and toggle the panorama.
                inputStatus["throttle"]   = stale ? 0.0f : joystickData.linear.z;
                inputStatus["yaw"]        = stale ? 0.0f : joystickData.angular.z;
                inputStatus["pitch"]      = stale ? 0.0f : joystickData.linear.x;
                inputStatus["roll"]       = stale ? 0.0f : joystickData.linear.y;
                // Under the adaptive spread the swarm sets its own spread about the scene's d_ref, so the spread
                // dial is ignored: written into d_ref, it replaced the value the scene was tuned at the moment the
                // controller connected (the bridge's knob starts at 1.0; BlockShapedPatchCityWorld flies 2.0).
                // -1 is "no override", which the recorder logs and analyse.py reads as the scene's d_ref. In ANY
                // mode the keyboard's A/D value, written above, stands.
                bool adaptiveSpread = SwarmManager.Instance != null && SwarmManager.Instance.GetAdaptiveSpread();
                if (!adaptiveSpread)
                    inputStatus["spread"] = joystickData.angular.x;
                else if (inputMode == InputMode.JOYSTICK)
                    inputStatus["spread"] = -1.0f;
                inputStatus["userSwitch"] = joystickData.switches.s1;
                inputStatuRaw = new Dictionary<string, float>(inputStatus);

                // Pitch dial (s2, raw -1..1) drives the swarm-wide FPV gimbal pitch, but only when it moves: the
                // first value is a baseline, so the scene's (or the inspector's) pitch holds until the pilot turns
                // the dial, and the inspector can still set it between turns.
                float dial = joystickData.switches.s2;
                if (!stale)
                {
                    if (!float.IsNaN(gimbalDial) && dial != gimbalDial && SwarmManager.Instance != null)
                        SwarmManager.Instance.SetGimbalPitchNormalized(dial);
                    gimbalDial = dial;
                }
            }
        }
    }

    public void LockControl()
    {
        isControlLocked = true;
    }
    public void UnlockControl()
    {
        isControlLocked = false;
    }

}