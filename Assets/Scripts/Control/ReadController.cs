using UnityEngine;

public class ReadController : MonoBehaviour
{

    public VelocityControl vc;
    public OlfatiSaber olfatiSaber;

    // Use the SwarmAlgorithm enum from SwarmManager
    public SwarmManager.SwarmAlgorithm currentAlgorithm;

    // Current algorithm selected
    private SwarmManager.AttitudeControl attitudeControlType;
    
    private float abs_height = 1;

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
                    olfatiSaber.desired_vx = linearVelocity.x;
                    olfatiSaber.desired_vy = linearVelocity.y;
                    olfatiSaber.desired_yaw = angularVelocity.z;
                    olfatiSaber.desired_height = abs_height;
                }

            }
        }
    }
}
