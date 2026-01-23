using UnityEngine;

public class ReadController : MonoBehaviour
{

    public VelocityControl vc;
    public SwarmAlgorithm SwarmAlgorithm;

    // Thrust modes
    public enum ThrustMode
    {
        HEIGHT_VARIABLE = -1,
        HEIGHT_CONSTANT
    }

    // View modes
    public enum ViewMode
    {
        FRONT = -1,
        BIRDSEYE,
        BACK
    }

    public float height_min = 10;
    public float height_max = 20;
    public SwarmAlgorithm SwarmAlgorithm;

    // Thrust modes
    public enum ThrustMode
    {
        HEIGHT_VARIABLE = -1,
        HEIGHT_CONSTANT
    }

    // View modes
    public enum ViewMode
    {
        FRONT = -1,
        BIRDSEYE,
        BACK
    }

    public float height_min = 10;
    public float height_max = 20;

    // Use the SwarmAlgorithm enum from SwarmManager
    public SwarmManager.SwarmAlgorithm currentAlgorithm;


    private float abs_height = 14;

    private float abs_height = 14;

    void Start()
    {
        
        
    }

    void Update()
    {
        // Check if there's any shared joystick data available
        if (UDPReceiverManager.sharedJoystickData != null)
        {
            // Access the shared joystick data from the central UDPReceiverManager
            JoystickData joystickData = UDPReceiverManager.sharedJoystickData;

            if (joystickData != null)
            {
                // Apply linear velocity to this object
                Vector3 linearVelocity = new Vector3(joystickData.linear.x, joystickData.linear.y, joystickData.linear.z);

                // Apply angular velocity to this object
                Vector3 angularVelocity = new Vector3(joystickData.angular.x, joystickData.angular.y, joystickData.angular.z);
                
                
                if (currentAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS)
                {
                    // Apply the velocities to the velocity control script
                    vc.desired_vx = linearVelocity.x;
                    vc.desired_vy = linearVelocity.y;
                    vc.desired_yaw = angularVelocity.z;
                    abs_height += linearVelocity.z * 0.1f;
                    vc.desired_height = abs_height;
                }
                else if (currentAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER)
                {
                    // Apply the velocities to the Olfati-Saber velocity control script
                    SwarmAlgorithm.desired_vx = linearVelocity.x * 4.0f;
                    SwarmAlgorithm.desired_vy = linearVelocity.y * 4.0f;
                    SwarmAlgorithm.desired_yaw = angularVelocity.z;
                    SwarmAlgorithm.SetSwarmSpread(angularVelocity.x);
                    
                    switch ((ThrustMode)joystickData.switches.s1)
                    {
                        case ThrustMode.HEIGHT_CONSTANT:
                            abs_height = height_min + (height_max - height_min) * (linearVelocity.z + 1.0f) / 2;
                            break;
                        case ThrustMode.HEIGHT_VARIABLE:
                            vc.swarm_vy = linearVelocity.z;
                            break;
                    }
                    //SwarmAlgorithm.desired_height = abs_height;

                    // Debug.Log($"Olfati desired vx: {linearVelocity.x}");
                    // Debug.Log($"Olfati desired vy: {linearVelocity.y}");
                    // Debug.Log($"Olfati desired yaw: {angularVelocity.z}");
                    // Debug.Log($"Olfati desired height: {abs_height}");
                    // Debug.Log($"Olfati switch: {joystickData.switches.s1}");
                }

            }
        }
    }
}
