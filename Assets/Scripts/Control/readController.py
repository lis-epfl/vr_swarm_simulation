import pygame
import socket
import json
import time

# Initialize pygame and joystick
pygame.init()
pygame.joystick.init()

# Create a UDP socket
udp_ip = "127.0.0.1"  # Change to the IP address of your Unity instance, if necessary
udp_port = 5055        # Change this to the port Unity is listening on

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Function to filter angular values (simulating your `filter` function)
def filter(value, current):
    # You can implement your own filtering logic here
    return value  # Just passing through for now

# Function to get joystick data (focused on the relevant axes)
def get_joystick_data(joystick, center_ang_x, current_ang_x):
    data = {}

    # Get specific axes data (from the controller mapping)
    data['linear'] = {
        'x': joystick.get_axis(3),  # Axis 2: linear.x
        'y': -joystick.get_axis(2),  # Axis 1: linear.y (negated)
        'z': 0.5 * joystick.get_axis(1) * abs(joystick.get_axis(1)),  # Axis 0: quadratic scaled linear.z
    }

    # Angular velocities
    data['angular'] = {
        'x': filter(center_ang_x + joystick.get_axis(4) * 0.4, current_ang_x),  # Axis 5: angular.x with filtering
        'y': 0.0,  # angular.y is fixed
        'z': -joystick.get_axis(0),  # Axis 3: angular.z (negated)
    }

    return data

# Main loop
center_ang_x = 1.0  # Simulated center for angular.x (adjust based on your needs)
current_ang_x = 1.0  # Simulated current angular.x value

try:
    while True:
        pygame.event.pump()  # Process event queue to update the joystick state

        # Assuming we're using the first joystick
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()

            # Get joystick data for the specific axes
            joystick_data = get_joystick_data(joystick, center_ang_x, current_ang_x)

            # Print the data for debugging
            # print(joystick_data)
            
            # Convert data to JSON string
            message = json.dumps(joystick_data)
            
            # Send the data over UDP
            sock.sendto(message.encode('utf-8'), (udp_ip, udp_port))
        
        # Sleep to limit the frequency of messages
        time.sleep(0.05)

except KeyboardInterrupt:
    # Clean up on exit
    pygame.quit()
    sock.close()
