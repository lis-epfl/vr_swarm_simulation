using UnityEngine;

public class ReadController : MonoBehaviour
{

    public VelocityControl vc;
    public OlfatiSaber olfatiSaber;
    
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
                
                // Apply the velocities to the velocity control script
                vc.desired_vx = linearVelocity.x;
                vc.desired_vy = linearVelocity.y;
                vc.desired_yaw = angularVelocity.z;
                abs_height += linearVelocity.z * 0.1f;
                vc.desired_height = abs_height;

                // Apply the velocities to the Olfati-Saber velocity control script
                olfatiSaber.desired_vx = linearVelocity.x;
                olfatiSaber.desired_vy = linearVelocity.y;
                olfatiSaber.desired_yaw = angularVelocity.z;
                olfatiSaber.desired_height = abs_height;

            }
        }
    }
}
