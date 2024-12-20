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
    
    def detect_board(self, image_path: str) -> Optional[List[List[StoneColor]]]:
        """
        Detect Go board and stones using YOLOv8
        Returns a 19x19 grid of stone positions
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
            
            # Process detections
            board = [[StoneColor.EMPTY for _ in range(self.board_size)] 
                    for _ in range(self.board_size)]
                    
            if len(results) == 0:
                logger.warning("No detections found")
                return None
                
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Process each detection
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
            
            return board
            
        except Exception as e:
            logger.error(f"Error in board detection: {str(e)}")
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
