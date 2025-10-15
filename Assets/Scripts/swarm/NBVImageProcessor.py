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
            command_memory_size = self.COMMAND_DATA_POSITION + 12  # 3 floats
            
            # Open shared memory maps
            self.image_mmf = mmap.mmap(-1, total_image_memory_size, self.image_memory_name)
            self.command_mmf = mmap.mmap(-1, command_memory_size, self.command_memory_name)
            
            if self.debug:
                print(f"✅ NBVImageProcessor initialized successfully")
                print(f"   Image memory: {total_image_memory_size} bytes")
                print(f"   Command memory: {command_memory_size} bytes")
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
    
    def process_images(self, image_data: ImageData) -> Optional[PositionCommand]:
        """
        Process drone images and determine position command
        
        This is where you would implement your computer vision algorithms.
        For now, it's a placeholder that demonstrates the interface.
        
        Args:
            image_data: ImageData containing drone images
            
        Returns:
            PositionCommand or None
        """
        if not image_data.images:
            return None
            
        try:
            # === PLACEHOLDER PROCESSING ===
            # This is where you would implement your actual OpenCV processing
            # Examples:
            # - Object detection
            # - Feature matching
            # - Optical flow
            # - Obstacle detection
            # - Path planning
            
            # For demonstration, let's do some basic analysis
            # total_brightness = 0
            # edge_density = 0
            
            # for i, image in enumerate(image_data.images):
            #     # Convert to grayscale for analysis
            #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
            #     # Calculate brightness
            #     brightness = np.mean(gray)
            #     total_brightness += brightness
                
            #     # Calculate edge density
            #     edges = cv2.Canny(gray, 50, 150)
            #     edge_pixels = np.sum(edges > 0)
            #     edge_density += edge_pixels / (gray.shape[0] * gray.shape[1])
                
            #     if self.debug and i == 0:  # Only log for first image
            #         print(f"🖼️ Drone {i}: Brightness={brightness:.1f}, Edges={edge_pixels}")
            
            # avg_brightness = total_brightness / len(image_data.images)
            # avg_edge_density = edge_density / len(image_data.images)
            
            # === RANDOM HEIGHT TESTING LOGIC ===
            # For visual confirmation that the system works
            # Generate random height adjustments every processing interval
            
            # Smaller movements to prevent runaway behavior
            import random
            x_movement = random.uniform(-1.0, 1.0)  # Small horizontal movement
            z_movement = random.uniform(-1.0, 1.0)  # Small forward/backward movement
            
            # Random height adjustment between -2 and +2 (reduced from -5 to +5)
            y_movement = random.uniform(-2.0, 2.0)
            
            if self.debug:
                print(f"🎲 Generated random command: X={x_movement:.2f}, Y={y_movement:.2f}, Z={z_movement:.2f}")
            
            # Original vision-based logic (commented out for testing)
            # if avg_brightness < 100:
            #     x_movement = 0.1  # Move right
            # elif avg_brightness > 150:
            #     x_movement = -0.1  # Move left
            # if avg_edge_density > 0.1:
            #     z_movement = 0.1  # Move up
            
            # For now, keep movements small for safety
            command = PositionCommand(x_movement, y_movement, z_movement)
            
            if self.debug:
                # print(f"🧠 Analysis: Brightness={avg_brightness:.1f}, EdgeDensity={avg_edge_density:.3f}")
                print(f"🎯 Command: Move ({command.x:.2f}, {command.y:.2f}, {command.z:.2f})")
            
            return command
            
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
                
                # Read images from shared memory (optional for testing)
                image_data = self.read_images_from_memory()
                
                # Generate position command regardless of image availability
                # === RANDOM HEIGHT TESTING LOGIC ===
                # send an x and z random movement between 10 and -10
                import random
                x_movement = random.uniform(-10.0, 10.0)
                z_movement = random.uniform(-10.0, 10.0)

                # Random height adjustment between -10 and +10 (Y = up/down in Unity)
                y_movement = random.uniform(-10.0, 10.0)

                command = PositionCommand(x_movement, y_movement, z_movement)
                
                if self.debug:
                    print(f"🎲 Generated random command: Y={y_movement:.2f}")
                
                # Send command back to Unity
                success = self.send_position_command(command)
                
                if success:
                    if self.debug:
                        print(f"✅ Command sent successfully")
                else:
                    if self.debug:
                        print(f"❌ Failed to send command")
                
                self.images_processed += 1
                self.last_processed_time = current_time
                
                if self.debug and self.images_processed % 5 == 0:
                    print(f"📊 Processed {self.images_processed} cycles, sent {self.commands_sent} commands")
                
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
    
    # Create processor with faster processing interval for testing
    processor = NBVImageProcessor(
        processing_interval=5.0,  # Process every 5 seconds (faster for testing)
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