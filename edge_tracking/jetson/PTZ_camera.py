import cv2
import subprocess
import time
import numpy as np
import os
import fcntl
import v4l2

from action_classifier import ActionClassifier
from person_tracker import PersonTracker


class PTZCamera:
    def __init__(self):
        self.device = '/dev/video0'

        # Control IDs from v4l2-ctl output
        self.ZOOM_ABSOLUTE = 0x009a090d
        self.PAN_SPEED = 0x009a0920
        self.TILT_SPEED = 0x009a0921

        # Open the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")

        # Set resolution to 640x480 for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Set frame rate to 30 fps
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Reduce buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialize person tracker
        self.person_tracker = PersonTracker(max_disappeared=30, max_distance=100)

        # Initialize YOLO11 model
        self.person_tracker.initialize_model()

        # Initialize action classifier
        self.action_classifier = ActionClassifier()

        # Tracking parameters
        self.tracking_enabled = False
        self.target_id = None
        self.frame_center = np.array([320, 240])  # Center of 640x480 frame
        self.deadzone = 50  # Pixels from center where no movement is needed

        # FPS calculation variables
        self.fps = 0
        self.prev_frame_time = 0
        self.curr_frame_time = 0

        # Print actual camera settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera initialized at {actual_width}x{actual_height} @ {actual_fps}fps")


    def set_control(self, control_id, value):
        """Set a V4L2 control using direct ioctl call"""
        try:

            # Open device if not already open
            if not hasattr(self, 'control_fd'):
                self.control_fd = os.open(self.device, os.O_RDWR)

            # Create control structure
            control = v4l2.v4l2_control()
            control.id = control_id
            control.value = int(value)

            # Direct ioctl call instead of subprocess
            fcntl.ioctl(self.control_fd, v4l2.VIDIOC_S_CTRL, control)
            return True

        except Exception as e:
            # Fallback to subprocess if direct method fails
            try:
                cmd = ['v4l2-ctl', '-d', self.device, '-c', f'{hex(control_id)}={value}']
                subprocess.run(cmd, check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error setting control {hex(control_id)}: {e}")
                return False

    def set_pan_speed(self, value):
        """Set pan speed (-1 to 1)"""
        speed = int(value)
        return self.set_control(self.PAN_SPEED, speed)

    def set_tilt_speed(self, value):
        """Set tilt speed (-1 to 1)"""
        speed = int(value)
        return self.set_control(self.TILT_SPEED, speed)

    def set_zoom(self, value):
        """Set zoom level (100 to 1000)"""
        zoom = int(100 + (value * (900/255)))
        return self.set_control(self.ZOOM_ABSOLUTE, zoom)

    def calculate_movement(self, target_center):
        """Calculate required pan/tilt movement based on target position"""
        error = target_center - self.frame_center

        # Don't move if target is within deadzone
        if abs(error[0]) < self.deadzone:
            pan = 0
        else:
            pan = 1 if error[0] > 0 else -1

        if abs(error[1]) < self.deadzone:
            tilt = 0
        else:
            tilt = -1 if error[1] > 0 else 1

        return pan, tilt

    def pan_search_for_target(self):
        """
        Performs a one-time sweep from left to right when target is lost.
        Returns True if search is in progress, False when completed.
        """
        # Initialize class variables if they don't exist
        if not hasattr(self, 'current_search_phase'):
            self.current_search_phase = "initial"

        if not hasattr(self, 'target_lost_time'):
            self.target_lost_time = None

        if not hasattr(self, 'search_start_time'):
            self.search_start_time = 0

        if not hasattr(self, 'search_duration'):
            self.search_duration = 16  # Time in seconds for a complete pan

        current_time = time.time()

        if self.current_search_phase == "initial":
            self.set_pan_speed(0)
            self.current_search_phase = "idle"

        # Check if we need to wait before starting the search
        if self.current_search_phase == "idle":
            # Initialize the target lost time if it's not set
            if self.target_lost_time is None:
                self.target_lost_time = current_time
                print("Target lost - waiting 5 seconds before searching...")

            # Wait 5 seconds before starting the search
            if current_time - self.target_lost_time < 5.0:
                # Still waiting
                print(f"Target lost for {current_time - self.target_lost_time}")

            # Start
            self.search_start_time = current_time
            self.current_search_phase = "wait"
            print("Target lost - starting search from left to right")

        elif self.current_search_phase != "complete":
            current_time = time.time()
            elapsed = current_time - self.search_start_time

            # First second: move to leftmost position
            if elapsed < 8.0:
                if self.current_search_phase != "left":
                    print("Turning left....")
                    self.set_pan_speed(-1)
                    self.current_search_phase = "left"


            # Next several seconds: pan right (from 5 seconds to search_duration)
            elif elapsed > 8.0 and elapsed <= self.search_duration:
                if self.current_search_phase != "right":
                    print("Turning right....")
                    self.set_pan_speed(1)
                    self.current_search_phase = "right"

            # Search complete
            else:
                if self.current_search_phase != "complete":
                    print("Search complete - target not found")
                    self.set_pan_speed(0)
                    self.current_search_phase = "complete"