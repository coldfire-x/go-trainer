import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from ..models.go_board import StoneColor
import logging

logger = logging.getLogger(__name__)

class BoardRecognitionService:
    def __init__(self):
        self.board_size = 19
        self.min_stone_area = 100  # Minimum area for a stone
        self.stone_circularity_threshold = 0.6  # Threshold for stone shape
        self.grid_threshold = 50  # Threshold for grid line detection
        
    def detect_board(self, image_path: str) -> Optional[List[List[StoneColor]]]:
        """
        Detect Go board and stones from an image
        Returns a 19x19 grid of stone positions
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image from {image_path}")
                return None
                
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            processed = self._preprocess_image(gray)
            
            # Detect grid lines
            grid_lines = self._detect_grid_lines(processed)
            if not grid_lines:
                logger.warning("No grid lines detected")
                return None
                
            # Find grid intersections
            intersections = self._find_intersections(grid_lines)
            if len(intersections) < (self.board_size * self.board_size) / 2:
                logger.warning(f"Too few intersections found: {len(intersections)}")
                return None
                
            # Create transformation matrix
            board_points = self._get_board_points(intersections)
            if board_points is None:
                logger.warning("Failed to get board points")
                return None
                
            # Initialize empty board
            board = [[StoneColor.EMPTY for _ in range(self.board_size)] 
                    for _ in range(self.board_size)]
            
            # Detect stones
            stones = self._detect_stones(gray)
            
            # Map stones to board positions
            for stone in stones:
                board_x, board_y = self._map_to_board_coordinates(
                    stone['center'], width, height
                )
                if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                    board[board_y][board_x] = stone['color']
            
            return board
            
        except Exception as e:
            logger.error(f"Error in board detection: {str(e)}")
            return None
    
    def _preprocess_image(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess image for better detection"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def _detect_grid_lines(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect grid lines using Hough transform"""
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=self.grid_threshold,
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is None:
            return []
            
        return [(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in [line[0]]]
    
    def _find_intersections(
        self, lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int]]:
        """Find grid intersections"""
        intersections = []
        for i, (x1, y1, x2, y2) in enumerate(lines):
            for x3, y3, x4, y4 in lines[i+1:]:
                # Line intersection formula
                denominator = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
                if denominator == 0:
                    continue
                    
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
                
                if 0 <= t <= 1 and 0 <= u <= 1:
                    x = int(x1 + t * (x2 - x1))
                    y = int(y1 + t * (y2 - y1))
                    intersections.append((x, y))
        
        return intersections
    
    def _get_board_points(
        self, intersections: List[Tuple[int, int]]
    ) -> Optional[np.ndarray]:
        """Get board corner points from intersections"""
        if not intersections:
            return None
            
        # Convert to numpy array
        points = np.array(intersections)
        
        # Find corner points
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        
        return np.float32(box)
    
    def _detect_stones(self, gray: np.ndarray) -> List[Dict]:
        """Detect stones on the board"""
        stones = []
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < self.min_stone_area:
                continue
                
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Check if the contour is roughly circular
            area = cv2.contourArea(contour)
            circle_area = np.pi * radius * radius
            if area / circle_area < self.stone_circularity_threshold:
                continue
            
            # Determine stone color
            roi = gray[
                max(0, center[1] - radius):min(gray.shape[0], center[1] + radius),
                max(0, center[0] - radius):min(gray.shape[1], center[0] + radius)
            ]
            if roi.size > 0:
                mean_intensity = np.mean(roi)
                color = StoneColor.BLACK if mean_intensity < 128 else StoneColor.WHITE
                stones.append({
                    'center': center,
                    'radius': radius,
                    'color': color
                })
        
        return stones
    
    def _map_to_board_coordinates(
        self, point: Tuple[int, int], width: int, height: int
    ) -> Tuple[int, int]:
        """Map image coordinates to board coordinates"""
        x, y = point
        board_x = int(x * self.board_size / width)
        board_y = int(y * self.board_size / height)
        return board_x, board_y
