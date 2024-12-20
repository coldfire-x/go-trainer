import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..models.go_board import StoneColor

class BoardRecognitionService:
    def __init__(self):
        self.board_size = 19
        
    def detect_board(self, image_path: str) -> Optional[List[List[StoneColor]]]:
        """
        Detect Go board and stones from an image
        Returns a 19x19 grid of stone positions
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Initialize empty board
        board = [[StoneColor.EMPTY for _ in range(self.board_size)] 
                for _ in range(self.board_size)]
        
        # Process each contour
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 100:
                continue
                
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Check if the contour is roughly circular
            area = cv2.contourArea(contour)
            circle_area = np.pi * radius * radius
            if area / circle_area < 0.6:
                continue
            
            # Determine stone color
            roi = gray[
                max(0, center[1] - radius):min(gray.shape[0], center[1] + radius),
                max(0, center[0] - radius):min(gray.shape[1], center[0] + radius)
            ]
            if roi.size > 0:
                mean_intensity = np.mean(roi)
                color = StoneColor.BLACK if mean_intensity < 128 else StoneColor.WHITE
                
                # Map to board coordinates
                board_x = int(x * self.board_size / img.shape[1])
                board_y = int(y * self.board_size / img.shape[0])
                
                if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                    board[board_y][board_x] = color
        
        return board
    
    def _detect_grid_lines(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect grid lines using Hough transform"""
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=100, 
            minLineLength=100, maxLineGap=10
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
