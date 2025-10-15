"""
NBVImageProcessor.py - Python counterpart to NBVImageCapture.cs

This script demonstrates the complete pipeline:
1. Reads drone images from shared memory (written by NBVImageCapture.cs)
2. Processes images using OpenCV (placeholder for now)
3. Sends position commands back to Unity via shared memory
4. Integrates with the existing StitcherThreading.py architecture

Usage:
    python NBVImageProcessor.py

Architecture:
- Uses memory-mapped files for real-time communication with Unity
- Runs at a slower frequency than image capture (configurable)
- Sends position deltas back to NBV.cs for drone movement
"""

import numpy as np
import cv2
import mmap
import struct
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ImageData:
    """Container for drone image data"""
    images: List[np.ndarray]
    timestamp: float
    image_count: int

@dataclass 
class PositionCommand:
    """Container for position command data"""
    x: float
    y: float
    z: float
    
    def to_bytes(self) -> bytes:
        """Convert to bytes for shared memory"""
        return struct.pack('fff', self.x, self.y, self.z)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'PositionCommand':
        """Create from bytes"""
        x, y, z = struct.unpack('fff', data)
        return cls(x, y, z)

class NBVImageProcessor:
    """
    Main processor class that handles:
    - Reading images from shared memory
    - Processing images with OpenCV
    - Sending position commands back to Unity
    """
    
    def __init__(self, 
                 processing_interval: float = 3.0,
                 debug: bool = True):
        """
        Initialize the image processor
        
        Args:
            processing_interval: How often to process images (seconds)
            debug: Enable debug logging
        """
        self.processing_interval = processing_interval
        self.debug = debug
        
        # Shared memory names (must match NBVImageCapture.cs)
        self.image_memory_name = "NBVImageSharedMemory"
        self.command_memory_name = "NBVCommandSharedMemory"
        
        # Memory layout constants
        self.FLAG_POSITION = 0
        self.IMAGE_COUNT_POSITION = 4
        self.IMAGE_DATA_POSITION = 8
        self.COMMAND_FLAG_POSITION = 0
        self.COMMAND_DATA_POSITION = 4
        
        # Per-drone command system
        self.per_drone_commands = True  # Enable individual drone commands
        self.command_size_per_drone = 12  # 3 floats (x, y, z)
        
        # Image settings (should match Unity settings)
        self.image_width = 300
        self.image_height = 300
        self.image_size = self.image_width * self.image_height * 3  # RGB
        self.max_drone_count = 10
        
        # Memory objects
        self.image_mmf: Optional[mmap.mmap] = None
        self.command_mmf: Optional[mmap.mmap] = None
        
        # Processing state
        self.running = False
        self.last_processed_time = 0
        self.current_position_command = PositionCommand(0, 0, 0)
        
        # Statistics
        self.images_processed = 0
        self.commands_sent = 0
        
    def initialize(self) -> bool:
        """
        Initialize shared memory connections
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate memory sizes
            total_image_memory_size = self.IMAGE_DATA_POSITION + (self.max_drone_count * self.image_size)
            
            if self.per_drone_commands:
                # Per-drone commands: one command per drone
                command_memory_size = self.COMMAND_DATA_POSITION + (self.max_drone_count * self.command_size_per_drone)
            else:
                # Single shared command
                command_memory_size = self.COMMAND_DATA_POSITION + 12  # 3 floats
            
            # Open shared memory maps
            self.image_mmf = mmap.mmap(-1, total_image_memory_size, self.image_memory_name)
            self.command_mmf = mmap.mmap(-1, command_memory_size, self.command_memory_name)
            
            if self.debug:
                print(f"✅ NBVImageProcessor initialized successfully")
                print(f"   Image memory: {total_image_memory_size} bytes")
                print(f"   Command memory: {command_memory_size} bytes")
                print(f"   Per-drone commands: {self.per_drone_commands}")
                print(f"   Processing interval: {self.processing_interval}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize shared memory: {e}")
            return False
    
    def read_images_from_memory(self) -> Optional[ImageData]:
        """
        Read drone images from shared memory
        
        Returns:
            ImageData object if successful, None otherwise
        """
        if not self.image_mmf:
            return None
            
        try:
            # Check flag - only read if Unity isn't writing (flag == 0)
            self.image_mmf.seek(self.FLAG_POSITION)
            flag = struct.unpack('i', self.image_mmf.read(4))[0]
            
            if flag != 0:
                return None  # Unity is writing, skip this read
            
            # Set flag to indicate we're reading
            self.image_mmf.seek(self.FLAG_POSITION)
            self.image_mmf.write(struct.pack('i', 1))
            
            try:
                # Read image count
                self.image_mmf.seek(self.IMAGE_COUNT_POSITION)
                image_count = struct.unpack('i', self.image_mmf.read(4))[0]
                
                if image_count <= 0:
                    return None
                
                # Read all images
                images = []
                for i in range(image_count):
                    # Calculate position for this image
                    image_position = self.IMAGE_DATA_POSITION + (i * self.image_size)
                    
                    # Read image data
                    self.image_mmf.seek(image_position)
                    image_data = self.image_mmf.read(self.image_size)
                    
                    # Convert to numpy array
                    image = np.frombuffer(image_data, dtype=np.uint8)
                    image = image.reshape((self.image_height, self.image_width, 3))
                    
                    # Unity uses different coordinate system, flip Y
                    image = cv2.flip(image, 0)
                    
                    images.append(image)
                
                return ImageData(
                    images=images,
                    timestamp=time.time(),
                    image_count=image_count
                )
                
            finally:
                # Always reset flag to indicate we're done reading
                self.image_mmf.seek(self.FLAG_POSITION)
                self.image_mmf.write(struct.pack('i', 0))
                
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error reading images from memory: {e}")
            return None
    
    def send_position_command(self, command: PositionCommand) -> bool:
        """
        Send position command to Unity via shared memory
        
        Args:
            command: PositionCommand to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.command_mmf:
            return False
            
        try:
            # Check if Unity is ready to receive (flag == 0)
            self.command_mmf.seek(self.COMMAND_FLAG_POSITION)
            flag = struct.unpack('i', self.command_mmf.read(4))[0]
            
            if self.debug:
                print(f"🔍 Current flag state: {flag}")
            
            if flag != 0:
                if self.debug:
                    print(f"⏳ Unity still processing previous command (flag={flag})")
                return False  # Unity is still processing previous command
            
            # Write position command first
            self.command_mmf.seek(self.COMMAND_DATA_POSITION)
            self.command_mmf.write(command.to_bytes())
            
            # Set flag to 2 to indicate new command is ready (not 1!)
            # This prevents race condition - Unity checks for >0, so 2 works
            # Unity will reset to 0 after reading, creating a handshake
            self.command_mmf.seek(self.COMMAND_FLAG_POSITION)
            self.command_mmf.write(struct.pack('i', 2))
            
            self.commands_sent += 1
            self.current_position_command = command
            
            if self.debug:
                print(f"📤 Sent position command: ({command.x:.2f}, {command.y:.2f}, {command.z:.2f}) [flag=2]")
            
            return True
                
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error sending position command: {e}")
            return False
    
    def send_per_drone_commands(self, commands: List[PositionCommand]) -> bool:
        """
        Send individual position commands for each drone
        
        Args:
            commands: List of PositionCommand objects, one per drone
            
        Returns:
            True if successful, False otherwise
        """
        if not self.command_mmf:
            return False
            
        try:
            # Check if Unity is ready to receive (flag == 0)
            self.command_mmf.seek(self.COMMAND_FLAG_POSITION)
            flag = struct.unpack('i', self.command_mmf.read(4))[0]
            
            if self.debug:
                print(f"🔍 Current flag state: {flag}")
            
            if flag != 0:
                if self.debug:
                    print(f"⏳ Unity still processing previous commands (flag={flag})")
                return False  # Unity is still processing previous commands
            
            # Write all drone commands
            for drone_id, command in enumerate(commands):
                if drone_id >= self.max_drone_count:
                    break
                    
                # Calculate position for this drone's command
                command_position = self.COMMAND_DATA_POSITION + (drone_id * self.command_size_per_drone)
                
                # Write command data
                self.command_mmf.seek(command_position)
                self.command_mmf.write(command.to_bytes())
                
                if self.debug and drone_id < 3:  # Log first 3 drones
                    print(f"📝 Drone {drone_id}: Command ({command.x:.2f}, {command.y:.2f}, {command.z:.2f}) at position {command_position}")
            
            # Set flag to 2 to indicate new commands are ready
            self.command_mmf.seek(self.COMMAND_FLAG_POSITION)
            self.command_mmf.write(struct.pack('i', 2))
            
            self.commands_sent += 1
            
            if self.debug:
                print(f"📤 Sent {len(commands)} per-drone commands [flag=2]")
            
            return True
                
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error sending per-drone commands: {e}")
            return False
    
    def process_images(self, image_data: ImageData) -> Optional[List[PositionCommand]]:
        """
        Process drone images and determine individual position commands per drone
        
        This is where you would implement your computer vision algorithms.
        Now generates individual commands for each drone based on its image.
        
        Args:
            image_data: ImageData containing drone images
            
        Returns:
            List of PositionCommand objects, one per drone, or None
        """
        if not image_data.images:
            return None
            
        try:
            commands = []
            
            # Process each drone's image individually
            for drone_id, image in enumerate(image_data.images):
                # === PER-DRONE PROCESSING ===
                # This is where you would implement your actual OpenCV processing
                # for each individual drone
                
                # For now, generate different random commands per drone
                import random
                
                # Each drone gets different movement ranges based on its ID
                # This makes it easy to see individual drone behavior
                movement_scale = (drone_id + 1) * 0.5  # Drone 0: 0.5x, Drone 1: 1.0x, etc.
                
                x_movement = random.uniform(-1.0, 1.0) * movement_scale
                z_movement = random.uniform(-1.0, 1.0) * movement_scale
                y_movement = random.uniform(-2.0, 2.0) * movement_scale
                
                # Create command for this specific drone
                command = PositionCommand(x_movement, y_movement, z_movement)
                commands.append(command)
                
                if self.debug and drone_id < 3:  # Log first 3 drones
                    print(f"🎲 Drone {drone_id}: Generated command X={x_movement:.2f}, Y={y_movement:.2f}, Z={z_movement:.2f} (scale={movement_scale:.1f})")
            
            # === FUTURE: Real Computer Vision Processing ===
            # for drone_id, image in enumerate(image_data.images):
            #     # Convert to grayscale for analysis
            #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            #     
            #     # Analyze this specific drone's image
            #     brightness = np.mean(gray)
            #     edges = cv2.Canny(gray, 50, 150)
            #     edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            #     
            #     # Generate movement command based on this drone's vision
            #     if brightness < 100:
            #         x_movement = 0.5  # Move toward brighter area
            #     elif brightness > 150:
            #         x_movement = -0.5  # Move away from bright area
            #     else:
            #         x_movement = 0
            #     
            #     if edge_density > 0.1:
            #         y_movement = 0.2  # Move up to get better view
            #     else:
            #         y_movement = -0.1  # Move down
            #         
            #     command = PositionCommand(x_movement, y_movement, 0)
            #     commands.append(command)
            
            if self.debug:
                print(f"🧠 Processed {len(commands)} drone images, generated individual commands")
            
            return commands
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error processing images: {e}")
            return None
    
    def processing_loop(self):
        """
        Main processing loop - runs in separate thread
        """
        self.running = True
        
        if self.debug:
            print(f"🚀 NBV Image Processing started (interval: {self.processing_interval}s)")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Only process at specified interval
                if current_time - self.last_processed_time < self.processing_interval:
                    time.sleep(0.1)
                    continue
                
                # For testing: always generate commands even without images
                if self.debug:
                    print(f"🔄 Processing cycle at {current_time:.1f}")
                
                # Read images from shared memory
                image_data = self.read_images_from_memory()
                
                if image_data is not None:
                    # Process images and get individual position commands per drone
                    position_commands = self.process_images(image_data)
                    
                    if position_commands is not None:
                        # Send per-drone commands back to Unity
                        if self.per_drone_commands:
                            success = self.send_per_drone_commands(position_commands)
                        else:
                            # Fallback to single command (use first drone's command)
                            success = self.send_position_command(position_commands[0]) if position_commands else False
                        
                        if success:
                            if self.debug:
                                print(f"✅ Commands sent successfully")
                        else:
                            if self.debug:
                                print(f"❌ Failed to send commands")
                    
                    self.images_processed += 1
                    self.last_processed_time = current_time
                    
                    if self.debug and self.images_processed % 5 == 0:
                        print(f"📊 Processed {self.images_processed} cycles, sent {self.commands_sent} commands")
                
                else:
                    # No new images available - for testing, still generate commands
                    if self.debug:
                        print(f"📷 No new images available, generating test commands...")
                    
                    # Generate test commands for available drones (assume 9 drones for testing)
                    test_commands = []
                    for drone_id in range(9):  # Adjust based on your drone count
                        import random
                        # Each drone gets different movement ranges
                        movement_scale = (drone_id + 1) * 5
                        x_movement = random.uniform(-1.0, 1.0) * movement_scale
                        z_movement = random.uniform(-1.0, 1.0) * movement_scale
                        y_movement = random.uniform(-2.0, 2.0) * movement_scale
                        
                        command = PositionCommand(x_movement, y_movement, z_movement)
                        test_commands.append(command)
                    
                    # Send test commands
                    if self.per_drone_commands:
                        success = self.send_per_drone_commands(test_commands)
                    else:
                        success = self.send_position_command(test_commands[0]) if test_commands else False
                    
                    if success:
                        if self.debug:
                            print(f"✅ Test commands sent successfully")
                    else:
                        if self.debug:
                            print(f"❌ Failed to send test commands")
                    
                    self.last_processed_time = current_time
                
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Error in processing loop: {e}")
                time.sleep(0.5)
    
    def start(self):
        """
        Start the processing in a separate thread
        """
        if not self.initialize():
            return False
            
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        
        return True
    
    def stop(self):
        """
        Stop the processing
        """
        self.running = False
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
        
        self.cleanup()
    
    def cleanup(self):
        """
        Clean up shared memory resources
        """
        if self.image_mmf:
            self.image_mmf.close()
            self.image_mmf = None
            
        if self.command_mmf:
            self.command_mmf.close()
            self.command_mmf = None
        
        if self.debug:
            print("🧹 NBVImageProcessor cleaned up")

def main():
    """
    Main function - demonstrates usage
    """
    print("🎮 NBV Image Processor - Real-time Drone Vision Processing")
    print("=" * 60)
    
    # Create processor with desired processing interval
    processor = NBVImageProcessor(
        processing_interval=3.0,  # Process every 3 seconds as requested
        debug=True
    )
    
    # Start processing
    if not processor.start():
        print("❌ Failed to start processor")
        return
    
    try:
        print("\n🔄 Processing started. Press Ctrl+C to stop.")
        print("📝 Make sure Unity is running with NBVImageCapture active!")
        print()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping processor...")
        processor.stop()
        print("✅ Processor stopped successfully")

if __name__ == "__main__":
    main()