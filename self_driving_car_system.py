import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from enum import Enum
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================
# CORE DATA STRUCTURES
# ============================

class VehicleState(Enum):
    STOPPED = "stopped"
    MOVING = "moving"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    EMERGENCY_BRAKE = "emergency_brake"
    PARKING = "parking"

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_name: str
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height

@dataclass
class LaneInfo:
    left_lane: Optional[np.ndarray]
    right_lane: Optional[np.ndarray]
    center_line: Optional[np.ndarray]
    lane_width: float
    offset_from_center: float
    curvature: float
    confidence: float

@dataclass
class VehicleControl:
    steering_angle: float  # -1.0 to 1.0 (left to right)
    throttle: float       # 0.0 to 1.0
    brake: float          # 0.0 to 1.0
    gear: str             # 'P', 'R', 'N', 'D'

@dataclass
class PerceptionData:
    timestamp: float
    detected_objects: List[BoundingBox]
    lane_info: LaneInfo
    traffic_signs: List[Dict]
    traffic_lights: List[Dict]
    speed_limit: Optional[int]
    road_conditions: Dict
    weather_conditions: Dict

# ============================
# CAMERA AND SENSOR MANAGEMENT
# ============================

class CameraManager:
    """Manage multiple cameras for 360-degree vision"""
    
    def __init__(self):
        self.cameras = {
            'front': None,
            'rear': None,
            'left': None,
            'right': None
        }
        self.camera_params = {
            'front': {'fov': 60, 'resolution': (1920, 1080)},
            'rear': {'fov': 120, 'resolution': (1280, 720)},
            'left': {'fov': 90, 'resolution': (1280, 720)},
            'right': {'fov': 90, 'resolution': (1280, 720)}
        }
        self.calibration_data = {}
    
    def initialize_cameras(self):
        """Initialize all camera feeds"""
        try:
            # For demo purposes, using webcam as front camera
            self.cameras['front'] = cv2.VideoCapture(0)
            if self.cameras['front'].isOpened():
                self.cameras['front'].set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cameras['front'].set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                logger.info("Front camera initialized successfully")
            else:
                logger.warning("Could not initialize front camera, using simulation mode")
                self.cameras['front'] = None
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
    
    def get_frame(self, camera_name: str) -> Optional[np.ndarray]:
        """Get frame from specified camera"""
        if camera_name not in self.cameras or self.cameras[camera_name] is None:
            # Return simulated frame for demo
            return self._generate_simulation_frame(camera_name)
        
        ret, frame = self.cameras[camera_name].read()
        return frame if ret else None
    
    def _generate_simulation_frame(self, camera_name: str) -> np.ndarray:
        """Generate simulated driving scene for demo"""
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create road background
        cv2.rectangle(frame, (0, height//2), (width, height), (50, 50, 50), -1)
        
        # Add lane markings
        for i in range(0, width, 60):
            cv2.rectangle(frame, (i, height//2 + 50), (i+30, height//2 + 60), (255, 255, 255), -1)
        
        # Add some "vehicles"
        if camera_name == 'front':
            # Add a car ahead
            cv2.rectangle(frame, (250, 200), (350, 280), (0, 0, 255), -1)
            cv2.putText(frame, "CAR", (270, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add traffic sign
            cv2.circle(frame, (500, 150), 30, (0, 255, 255), -1)
            cv2.putText(frame, "STOP", (480, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def release_cameras(self):
        """Release all camera resources"""
        for camera_name, camera in self.cameras.items():
            if camera is not None:
                camera.release()
        cv2.destroyAllWindows()

# ============================
# LANE DETECTION SYSTEM
# ============================

class AdvancedLaneDetector:
    """Advanced lane detection using perspective transform and polynomial fitting"""
    
    def __init__(self):
        self.calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Perspective transform matrices
        self.M = None
        self.Minv = None
        
        # Lane detection parameters
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50
        
        # Lane tracking
        self.left_fit_history = deque(maxlen=10)
        self.right_fit_history = deque(maxlen=10)
        
    def setup_perspective_transform(self, img_shape):
        """Setup perspective transform for bird's eye view"""
        height, width = img_shape[:2]
        
        # Define source points (trapezoid in original image)
        src = np.float32([
            [width * 0.15, height],           # Bottom left
            [width * 0.45, height * 0.6],    # Top left
            [width * 0.55, height * 0.6],    # Top right
            [width * 0.85, height]           # Bottom right
        ])
        
        # Define destination points (rectangle in bird's eye view)
        dst = np.float32([
            [width * 0.25, height],          # Bottom left
            [width * 0.25, 0],               # Top left
            [width * 0.75, 0],               # Top right
            [width * 0.75, height]           # Bottom right
        ])
        
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for lane detection"""
        # Convert to HLS color space
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        # Apply Sobel edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        # Create binary threshold for edges
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
        
        # Create binary threshold for color (yellow and white lines)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]
        
        # Yellow line detection
        yellow_binary = np.zeros_like(s_channel)
        yellow_binary[(s_channel >= 100) & (s_channel <= 255)] = 1
        
        # White line detection
        white_binary = np.zeros_like(l_channel)
        white_binary[(l_channel >= 200) & (l_channel <= 255)] = 1
        
        # Combine all binary images
        combined_binary = np.zeros_like(sobel_binary)
        combined_binary[(sobel_binary == 1) | (yellow_binary == 1) | (white_binary == 1)] = 1
        
        return combined_binary
    
    def detect_lanes(self, image: np.ndarray) -> LaneInfo:
        """Main lane detection function"""
        if self.M is None:
            self.setup_perspective_transform(image.shape)
        
        # Preprocess image
        binary_image = self.preprocess_image(image)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(binary_image, self.M, (image.shape[1], image.shape[0]))
        
        # Find lane pixels
        left_fitx, right_fitx, ploty = self._find_lane_pixels(warped)
        
        if left_fitx is not None and right_fitx is not None:
            # Fit polynomials
            left_fit = np.polyfit(ploty, left_fitx, 2)
            right_fit = np.polyfit(ploty, right_fitx, 2)
            
            # Store in history for smoothing
            self.left_fit_history.append(left_fit)
            self.right_fit_history.append(right_fit)
            
            # Calculate lane properties
            lane_width = np.mean(right_fitx - left_fitx)
            center_offset = self._calculate_center_offset(left_fitx, right_fitx, image.shape[1])
            curvature = self._calculate_curvature(left_fit, right_fit, ploty)
            
            # Generate lane lines for visualization
            left_lane = np.column_stack((left_fitx, ploty))
            right_lane = np.column_stack((right_fitx, ploty))
            center_line = np.column_stack(((left_fitx + right_fitx) / 2, ploty))
            
            return LaneInfo(
                left_lane=left_lane,
                right_lane=right_lane,
                center_line=center_line,
                lane_width=lane_width,
                offset_from_center=center_offset,
                curvature=curvature,
                confidence=0.9
            )
        
        return LaneInfo(None, None, None, 0, 0, 0, 0)
    
    def _find_lane_pixels(self, warped: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Find lane pixels using sliding window approach"""
        # Take histogram of bottom half
        histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)
        
        # Find peaks for left and right lanes
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Set height of windows
        window_height = warped.shape[0] // self.nwindows
        
        # Identify x and y positions of all nonzero pixels
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through windows
        for window in range(self.nwindows):
            # Identify window boundaries
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
            # Identify nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append indices to lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If found > minpix pixels, recenter next window
            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate arrays of indices
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            return None, None, None
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        if len(leftx) == 0 or len(rightx) == 0:
            return None, None, None
        
        # Generate y values for plotting
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        
        # Fit polynomials
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Calculate x values
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        return left_fitx, right_fitx, ploty
    
    def _calculate_center_offset(self, left_fitx: np.ndarray, right_fitx: np.ndarray, img_width: int) -> float:
        """Calculate offset from lane center"""
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
        image_center = img_width / 2
        
        # Convert to meters (assuming lane width is about 3.7 meters)
        meters_per_pixel = 3.7 / (right_fitx[-1] - left_fitx[-1])
        offset_meters = (lane_center - image_center) * meters_per_pixel
        
        return offset_meters
    
    def _calculate_curvature(self, left_fit: np.ndarray, right_fit: np.ndarray, ploty: np.ndarray) -> float:
        """Calculate radius of curvature"""
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        
        # Calculate new polynomials in world space
        y_eval = np.max(ploty)
        
        # Calculate curvature
        left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
        
        return (left_curverad + right_curverad) / 2

# ============================
# OBJECT DETECTION SYSTEM
# ============================

class YOLOObjectDetector:
    """YOLO-based object detection for vehicles, pedestrians, etc."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'
        ]
        
        # Relevant classes for driving
        self.driving_classes = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'bus': 5, 
            'train': 6, 'truck': 7, 'traffic light': 9, 'stop sign': 11
        }
        
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_simple_detector()
    
    def _initialize_simple_detector(self):
        """Initialize a simple CNN-based detector for demo"""
        self.model = SimpleObjectDetectionNet(len(self.driving_classes))
    
    def detect_objects(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect objects in image"""
        if self.model is None:
            return self._simple_detection_fallback(image)
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Run detection (simplified for demo)
        detections = self._simple_detection_fallback(image)
        
        return detections
    
    def _simple_detection_fallback(self, image: np.ndarray) -> List[BoundingBox]:
        """Simple detection using template matching and color detection"""
        detections = []
        
        # Detect red objects (potential vehicles/signs)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red color range
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify based on aspect ratio and position
                aspect_ratio = w / h if h > 0 else 0
                center_y = y + h // 2
                
                if aspect_ratio > 1.2 and center_y > image.shape[0] // 2:
                    class_name = 'car'
                elif aspect_ratio < 0.8 and area > 2000:
                    class_name = 'person'
                else:
                    class_name = 'object'
                
                confidence = min(area / 10000, 1.0)
                
                detections.append(BoundingBox(
                    x=x, y=y, width=w, height=h,
                    confidence=confidence, class_name=class_name
                ))
        
        return detections
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize to model input size
        resized = cv2.resize(image, (416, 416))
        
        # Convert to RGB and normalize
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor

class SimpleObjectDetectionNet(nn.Module):
    """Simplified object detection network for demo"""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes * 5)  # 5 = 4 bbox + 1 confidence
        
    def forward(self, x):
        return self.backbone(x)

# ============================
# TRAFFIC SIGN AND LIGHT DETECTION
# ============================

class TrafficSignDetector:
    """Detect and classify traffic signs and lights"""
    
    def __init__(self):
        self.sign_classifier = None
        self.traffic_light_states = ['red', 'yellow', 'green']
        
        # Traffic sign templates for matching
        self.sign_templates = {
            'stop': self._create_stop_sign_template(),
            'yield': self._create_yield_sign_template(),
            'speed_limit': self._create_speed_limit_template()
        }
    
    def detect_traffic_signs(self, image: np.ndarray) -> List[Dict]:
        """Detect traffic signs in image"""
        signs = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect red signs (stop, yield, speed limit)
        red_signs = self._detect_red_signs(hsv, image)
        signs.extend(red_signs)
        
        # Detect yellow warning signs
        yellow_signs = self._detect_yellow_signs(hsv, image)
        signs.extend(yellow_signs)
        
        return signs
    
    def detect_traffic_lights(self, image: np.ndarray) -> List[Dict]:
        """Detect traffic lights and determine state"""
        lights = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for traffic lights
        color_ranges = {
            'red': [(np.array([0, 50, 50]), np.array([10, 255, 255])),
                   (np.array([170, 50, 50]), np.array([180, 255, 255]))],
            'yellow': [(np.array([20, 50, 50]), np.array([30, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))]
        }
        
        for color, ranges in color_ranges.items():
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 100 < area < 5000:  # Appropriate size for traffic light
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Check if it's roughly circular (traffic light characteristic)
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.7 < aspect_ratio < 1.3:
                            lights.append({
                                'bbox': (x, y, w, h),
                                'state': color,
                                'confidence': min(area / 1000, 1.0)
                            })
        
        return lights
    
    def _detect_red_signs(self, hsv: np.ndarray, image: np.ndarray) -> List[Dict]:
        """Detect red traffic signs"""
        # Red color ranges
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        signs = []
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:  # Appropriate size for traffic sign
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify based on shape
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                sign_type = 'unknown'
                if len(approx) == 8:  # Octagon
                    sign_type = 'stop'
                elif len(approx) == 3:  # Triangle
                    sign_type = 'yield'
                elif 4 <= len(approx) <= 6:  # Rectangle/Pentagon
                    sign_type = 'speed_limit'
                
                signs.append({
                    'bbox': (x, y, w, h),
                    'type': sign_type,
                    'confidence': min(area / 5000, 1.0)
                })
        
        return signs
    
    def _detect_yellow_signs(self, hsv: np.ndarray, image: np.ndarray) -> List[Dict]:
        """Detect yellow warning signs"""
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        signs = []
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 8000:
                x, y, w, h = cv2.boundingRect(contour)
                
                signs.append({
                    'bbox': (x, y, w, h),
                    'type': 'warning',
                    'confidence': min(area / 4000, 1.0)
                })
        
        return signs
    
    def _create_stop_sign_template(self) -> np.ndarray:
        """Create stop sign template for matching"""
        template = np.zeros((60, 60, 3), dtype=np.uint8)
        # Draw octagon shape
        pts = np.array([[20, 10], [40, 10], [50, 20], [50, 40], 
                       [40, 50], [20, 50], [10, 40], [10, 20]], np.int32)
        cv2.fillPoly(template, [pts], (0, 0, 255))  # Red fill
        cv2.putText(template, "STOP", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return template
    
    def _create_yield_sign_template(self) -> np.ndarray:
        """Create yield sign template"""
        template = np.zeros((60, 60, 3), dtype=np.uint8)
        # Draw triangle
        pts = np.array([[30, 10], [10, 50], [50, 50]], np.int32)
        cv2.fillPoly(template, [pts], (0, 0, 255))
        cv2.putText(template, "YIELD", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return template
    
    def _create_speed_limit_template(self) -> np.ndarray:
        """Create speed limit sign template"""
        template = np.zeros((60, 60, 3), dtype=np.uint8)
        cv2.rectangle(template, (10, 10), (50, 50), (255, 255, 255