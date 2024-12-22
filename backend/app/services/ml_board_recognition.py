import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
from ..models.go_board import StoneColor
import logging
import os
import tempfile
from PIL import Image

logger = logging.getLogger(__name__)

class MLBoardRecognitionService:
    def __init__(self):
        self.board_size = 19
        self.model = self._load_model()
        
    def _load_model(self) -> YOLO:
        """Load or download YOLOv8 model"""
        try:
            # First try to load local model
            model_path = os.path.join(
                os.path.dirname(__file__), 
                "models", 
                "go_board_detector.pt"
            )
            if os.path.exists(model_path):
                return YOLO(model_path)
            
            # If local model doesn't exist, use pretrained model and fine-tune it
            logger.info("Local model not found, using pretrained model")
            model = YOLO('yolov8n.pt')
            
            # TODO: Fine-tune model on Go board dataset
            # This requires collecting and labeling Go board images
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def detect_board(self, image_path: str) -> Optional[Tuple[List[List[StoneColor]], str]]:
        """
        Detect Go board and stones using YOLOv8
        Returns: (board_state, corner_type)
        board_state: 19x19 grid of stone positions
        corner_type: which part of the board is visible
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image from {image_path}")
                return None
            
            # Convert to RGB for YOLO
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(img_rgb)
            
            # If ML detection fails, try traditional CV approach
            if len(results) == 0 or len(results[0].boxes.data) == 0:
                logger.warning("ML detection failed, falling back to traditional CV")
                return self._detect_board_cv(img)
            
            # Process detections
            board = [[StoneColor.EMPTY for _ in range(self.board_size)] 
                    for _ in range(self.board_size)]
                    
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Process each detection
            stones_detected = []
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = result
                
                # Skip low confidence detections
                if conf < 0.5:
                    continue
                    
                # Get center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Map to board coordinates
                board_x = int(center_x * self.board_size / width)
                board_y = int(center_y * self.board_size / height)
                
                # Determine stone color based on class
                color = StoneColor.BLACK if cls == 0 else StoneColor.WHITE
                
                if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                    board[board_y][board_x] = color
                    stones_detected.append((board_x, board_y))
            
            # If no stones were detected, try traditional CV approach
            if not stones_detected:
                logger.warning("No stones detected by ML, falling back to traditional CV")
                return self._detect_board_cv(img)
                
            # Determine corner type based on stone positions
            min_x = min(x for x, _ in stones_detected)
            max_x = max(x for x, _ in stones_detected)
            min_y = min(y for _, y in stones_detected)
            max_y = max(y for _, y in stones_detected)
            
            # Determine board section based on stone positions
            corner_type = 'middle'
            if min_x < 5 and min_y < 5:
                corner_type = 'top_left'
            elif max_x > 13 and min_y < 5:
                corner_type = 'top_right'
            elif min_x < 5 and max_y > 13:
                corner_type = 'bottom_left'
            elif max_x > 13 and max_y > 13:
                corner_type = 'bottom_right'
            
            return board, corner_type
            
        except Exception as e:
            logger.error(f"Error in ML board detection: {str(e)}")
            logger.info("Falling back to traditional CV")
            return self._detect_board_cv(img)
            
    def _detect_board_cv(self, img: np.ndarray) -> Optional[Tuple[List[List[StoneColor]], str]]:
        """Traditional CV approach for board detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding with smaller block size for better local contrast
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                7, 2  # Reduced block size from 11 to 7
            )
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
                
            # Find the largest contour (likely the board)
            board_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(board_contour)
            
            # Extract board region
            board_img = img[y:y+h, x:x+w]
            board_gray = gray[y:y+h, x:x+w]
            
            # Create empty board
            board = [[StoneColor.EMPTY for _ in range(self.board_size)] 
                    for _ in range(self.board_size)]
            
            # Calculate cell size
            cell_width = w / (self.board_size - 1)
            cell_height = h / (self.board_size - 1)
            
            # Detect stones using Hough circles
            circles = cv2.HoughCircles(
                board_gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min(cell_width, cell_height) * 0.8,  # Minimum distance between circles
                param1=50,
                param2=30,
                minRadius=int(min(cell_width, cell_height) * 0.2),
                maxRadius=int(min(cell_width, cell_height) * 0.5)
            )
            
            stones_detected = []
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    cx, cy, r = circle
                    
                    # Get the average intensity in the circle area
                    mask = np.zeros(board_gray.shape, np.uint8)
                    cv2.circle(mask, (cx, cy), r, 255, -1)
                    mean_val = cv2.mean(board_gray, mask=mask)[0]
                    
                    # Map to board coordinates
                    board_x = int(round(cx / cell_width))
                    board_y = int(round(cy / cell_height))
                    
                    if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                        # Determine stone color based on intensity
                        if mean_val < 128:  # Dark stone
                            board[board_y][board_x] = StoneColor.BLACK
                        else:  # Light stone
                            board[board_y][board_x] = StoneColor.WHITE
                        stones_detected.append((board_x, board_y))
            
            if not stones_detected:
                return None
                
            # Determine corner type based on stone positions
            min_x = min(x for x, _ in stones_detected)
            max_x = max(x for x, _ in stones_detected)
            min_y = min(y for _, y in stones_detected)
            max_y = max(y for _, y in stones_detected)
            
            # Calculate board section
            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            
            # Determine corner type based on center position
            corner_type = 'middle'
            if center_x < self.board_size * 0.4:
                if center_y < self.board_size * 0.4:
                    corner_type = 'top_left'
                elif center_y > self.board_size * 0.6:
                    corner_type = 'bottom_left'
            elif center_x > self.board_size * 0.6:
                if center_y < self.board_size * 0.4:
                    corner_type = 'top_right'
                elif center_y > self.board_size * 0.6:
                    corner_type = 'bottom_right'
            
            return board, corner_type
            
        except Exception as e:
            logger.error(f"Error in CV board detection: {str(e)}")
            return None
    
    def train_model(self, dataset_path: str):
        """
        Train the model on a custom Go board dataset
        dataset_path should point to a directory with images and YOLO format labels
        """
        try:
            # Configure training parameters
            self.model.train(
                data=dataset_path,
                epochs=100,
                imgsz=640,
                batch=16,
                name='go_board_detector'
            )
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better detection"""
        # Resize image to model input size
        return cv2.resize(img, (640, 640))
    
    def _postprocess_detections(
        self, 
        detections: List[Dict],
        original_size: Tuple[int, int]
    ) -> List[Dict]:
        """Scale detection coordinates back to original image size"""
        height, width = original_size
        processed = []
        
        for det in detections:
            # Scale coordinates
            x1 = det['x1'] * width / 640
            y1 = det['y1'] * height / 640
            x2 = det['x2'] * width / 640
            y2 = det['y2'] * height / 640
            
            processed.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'confidence': det['confidence'],
                'class': det['class']
            })
            
        return processed
