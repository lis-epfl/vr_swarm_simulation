using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputControl : MonoBehaviour {

    public VelocityControl vc;
    public OlfatiSaber olfatiSaber;

    public SwarmManager.SwarmAlgorithm currentAlgorithm;

    private float abs_height = 1;

    // Use this for initialization
    void Start () {
        
    }
    
    // Update is called once per frame
    void FixedUpdate () {
        
        
        if (currentAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS)
        {
            // Default velocity control
            vc.desired_vx = 0.0f; //Input.GetAxisRaw ("Pitch")*4.0f;
            vc.desired_vy = 0.0f; //Input.GetAxisRaw ("Roll")*4.0f;
            vc.desired_yaw = 0.0f; //Input.GetAxisRaw ("Yaw")*0.5f;
            abs_height += 0.0f; //Input.GetAxisRaw("Throttle") * 0.1f;
            // vc.desired_height = abs_height;
        }
        else if (currentAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER)
        {
            // FIXED: Use the calculated values from OlfatiSaber's hand-based calibration
            // Don't set to 0 - use the values calculated in UpdateVelocityFromHandPosition()
            
			// Pass the hand-calculated velocities from OlfatiSaber to VelocityControl
			// olfatiSaber.desired_vx=10.0f;
			vc.desired_vy = olfatiSaber.desired_vy;
			//vc.desired_vz = olfatiSaber.desired_vz; // Note: This might be mapped differently in VelocityControl
			vc.desired_yaw = 0.0f; // Keep yaw control as before
			
			// Debug to verify the values are being passed correctly
			// if (olfatiSaber.transform.parent.name == "Drone 0")
			// {
			// 	Debug.Log($"InputControl passing velocities - VX: {vc.desired_vx:F2}, VY: {olfatiSaber.desired_vy:F2}, VZ: {olfatiSaber.desired_vz:F2}");
			// }
            
        }
    }
}
