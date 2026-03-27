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
    [Header("Coefficients")]
    [SerializeField]
    private float speedCoeff = 4.0f;
    [SerializeField]
    private float yawRateCoeff = 1.0f;
    [SerializeField]
    private float altCoeff = 1.0f;

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
    public static InputManager Instance { get; private set; }
    public Dictionary<string, float> InputStatus => inputStatus; // Expose inputStatus as a read-only property
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
            return;
        }
        if (inputMode == InputMode.KEYBOARD || inputMode == InputMode.ANY )
        {
            // Update inputStatus based on keyboard input
            inputStatus["throttle"] = Input.GetAxisRaw("Throttle") * altCoeff;
            inputStatus["yaw"] = Input.GetAxisRaw("Yaw") * yawRateCoeff;
            inputStatus["pitch"] = Input.GetAxisRaw("Pitch") * speedCoeff;
            inputStatus["roll"] = Input.GetAxisRaw("Roll") * speedCoeff;
            inputStatus["userSwitch"] = Input.GetKey(KeyCode.Space) ? 1 : -1;
        }
        if (inputMode == InputMode.JOYSTICK || (inputMode == InputMode.ANY && !Input.anyKeyDown))
        {
            JoystickData joystickData = UDPReceiverManager.sharedJoystickData;
            if (joystickData != null)
            {
                inputStatus["throttle"] = joystickData.linear.z * altCoeff;
                inputStatus["yaw"] = joystickData.angular.z * yawRateCoeff;
                inputStatus["pitch"] = joystickData.linear.x * speedCoeff;
                inputStatus["roll"] = joystickData.linear.y * speedCoeff;
                inputStatus["spread"] = joystickData.angular.x; // Assuming spread is controlled by angular x
                inputStatus["userSwitch"] = joystickData.switches.s1; // Assuming buttons is a list of button names
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