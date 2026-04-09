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

    void Start()
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
            inputStatuRaw = new Dictionary<string, float>(inputStatus);
        }
        if (inputMode == InputMode.JOYSTICK || (inputMode == InputMode.ANY && !Input.anyKeyDown))
        {
            JoystickData joystickData = UDPReceiverManager.sharedJoystickData;
            if (joystickData != null)
            {
                inputStatus["throttle"]   = joystickData.linear.z;
                inputStatus["yaw"]        = joystickData.angular.z;
                inputStatus["pitch"]      = joystickData.linear.x;
                inputStatus["roll"]       = joystickData.linear.y;
                inputStatus["spread"]     = joystickData.angular.x;
                inputStatus["userSwitch"] = joystickData.switches.s1;
                inputStatuRaw = new Dictionary<string, float>(inputStatus);
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