import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from ..models.go_board import StoneColor
import logging

logger = logging.getLogger(__name__)

class BoardRecognitionService:
    def __init__(self):
        self.board_size = 19
        self.cell_size = 30  # Target size of each cell in pixels
        self.board_margin = 40  # Margin around the board
        self.stone_size = None  # Will be calibrated from black stones
        self.min_grid_lines = 4  # Minimum number of grid lines to detect a valid board section
        self.min_stones = 2  # Minimum number of stones needed for calibration
        
        # Star point positions on 19x19 board
        self.star_points = [
            (3, 3), (3, 9), (3, 15),
            (9, 3), (9, 9), (9, 15),
            (15, 3), (15, 9), (15, 15)
        ]
    
    def detect_board(self, image_path: str) -> Tuple[Optional[List[List[StoneColor]]], Optional[str]]:
        """
        Detect Go board and stones from an image
        Returns: (board_state, corner_type)
        board_state: 19x19 grid of stone positions
        corner_type: which part of the board is visible ('top_left', 'top_right', 'bottom_left', 'bottom_right', 'middle')
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image from {image_path}")
                return None, None
            
            # Preprocess image
            processed = self._preprocess_image(img)
            
            # Find grid lines and estimate board section
            grid_info = self._detect_grid_section(processed)
            if grid_info is None:
                logger.error("Failed to detect grid section")
                return None, None
            
            board_region, grid_lines, estimated_position, corner_type = grid_info
            
            # Extract and transform board region
            warped = self._get_board_transform(processed[board_region[1]:board_region[3], 
                                                       board_region[0]:board_region[2]], 
                                             img[board_region[1]:board_region[3], 
                                                 board_region[0]:board_region[2]])
            if warped is None:
                logger.error("Failed to transform board region")
                return None, None
            
            # Calibrate stone size using black stones
            self._calibrate_stone_size(warped)
            if self.stone_size is None:
                logger.warning("Failed to calibrate stone size")
                self.stone_size = int(min(warped.shape) / (self.board_size * 2))  # Fallback
            
            # Create grid cells for the visible section
            cells = self._create_grid_cells(warped)
            
            # Initialize empty board
            board = [[StoneColor.EMPTY for _ in range(self.board_size)] 
                    for _ in range(self.board_size)]
            
            # Map detected region to full board coordinates
            start_x, start_y = estimated_position
            visible_width = min(len(cells[0]), self.board_size - start_x)
            visible_height = min(len(cells), self.board_size - start_y)
            
            # Analyze visible cells and map to correct board positions
            for y in range(visible_height):
                for x in range(visible_width):
                    board_x = start_x + x
                    board_y = start_y + y
                    if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                        stone_color = self._analyze_cell(cells[y][x], board_x, board_y)
                        board[board_y][board_x] = stone_color
            
            return board, corner_type
            
        except Exception as e:
            logger.error(f"Error in board detection: {str(e)}")
            return None, None
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better board detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Use morphological operations to close gaps in lines
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def _get_board_transform(self, processed: np.ndarray, original: np.ndarray) -> Optional[np.ndarray]:
        """Find board corners and apply perspective transform"""
        # Find contours
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Find the largest contour (should be the board)
        board_contour = max(contours, key=cv2.contourArea)
        
        # Get approximate polygon
        epsilon = 0.02 * cv2.arcLength(board_contour, True)
        approx = cv2.approxPolyDP(board_contour, epsilon, True)
        
        # If we don't get exactly 4 corners, try to find the best 4
        if len(approx) != 4:
            rect = cv2.minAreaRect(board_contour)
            approx = cv2.boxPoints(rect)
        
        # Order points
        ordered_points = self._order_points(approx)
        
        # Calculate target size
        target_size = self.cell_size * (self.board_size - 1) + self.board_margin * 2
        
        # Define target points
        target_points = np.array([
            [0, 0],
            [target_size - 1, 0],
            [target_size - 1, target_size - 1],
            [0, target_size - 1]
        ], dtype=np.float32)
        
        # Get transform matrix
        transform = cv2.getPerspectiveTransform(ordered_points, target_points)
        
        # Apply transform
        warped = cv2.warpPerspective(original, transform, (target_size, target_size))
        
        return warped
    
    def _create_grid_cells(self, warped: np.ndarray) -> List[List[np.ndarray]]:
        """Create grid of cells from warped image"""
        cells = []
        height, width = warped.shape[:2]
        
        # Calculate cell dimensions
        cell_width = (width - 2 * self.board_margin) / (self.board_size - 1)
        cell_height = (height - 2 * self.board_margin) / (self.board_size - 1)
        
        # Create slightly larger cells to ensure stone detection
        cell_margin = min(cell_width, cell_height) * 0.4
        
        for y in range(self.board_size):
            row = []
            for x in range(self.board_size):
                # Calculate cell coordinates
                center_x = int(self.board_margin + x * cell_width)
                center_y = int(self.board_margin + y * cell_height)
                
                # Extract cell with margin
                x1 = max(0, int(center_x - cell_margin))
                y1 = max(0, int(center_y - cell_margin))
                x2 = min(width, int(center_x + cell_margin))
                y2 = min(height, int(center_y + cell_margin))
                
                cell = warped[y1:y2, x1:x2]
                row.append(cell)
            cells.append(row)
        
        return cells
    
    def _calibrate_stone_size(self, warped: np.ndarray) -> None:
        """
        Calibrate stone size by analyzing black stones in the image.
        Uses contour detection to find circular objects and estimates stone size.
        """
        try:
            if warped is None:
                return None
            
            # Convert to grayscale if needed
            if len(warped.shape) == 3:
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            else:
                gray = warped
            
            # Apply adaptive thresholding to handle varying lighting
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Filter contours by circularity and area
            stone_sizes = []
            min_circularity = 0.7
            cell_area = ((warped.shape[0] - 2 * self.board_margin) / (self.board_size - 1)) ** 2
            min_area = cell_area * 0.1  # Minimum stone area
            max_area = cell_area * 0.8  # Maximum stone area
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > min_circularity:
                    # Get the minimum enclosing circle
                    (_, _), radius = cv2.minEnclosingCircle(contour)
                    stone_sizes.append(radius * 2)  # Diameter
            
            if stone_sizes:
                # Use median of detected sizes to be robust against outliers
                self.stone_size = int(np.median(stone_sizes))
                # Validate the size is reasonable
                cell_size = (warped.shape[0] - 2 * self.board_margin) / (self.board_size - 1)
                if self.stone_size > cell_size * 0.9:  # Stone too large
                    self.stone_size = int(cell_size * 0.8)
                elif self.stone_size < cell_size * 0.3:  # Stone too small
                    self.stone_size = int(cell_size * 0.6)
            else:
                # Fallback to estimation based on cell size
                cell_size = (warped.shape[0] - 2 * self.board_margin) / (self.board_size - 1)
                self.stone_size = int(cell_size * 0.6)
            
            logger.info(f"Calibrated stone size: {self.stone_size}")
            
        except Exception as e:
            logger.error(f"Error in stone size calibration: {str(e)}")
            # Fallback to default size
            self.stone_size = int(min(warped.shape) / (self.board_size * 2))

    def _analyze_cell(self, cell: np.ndarray, board_x: int, board_y: int) -> StoneColor:
        """
        Analyze a cell to determine if it contains a stone.
        Uses both intensity and shape analysis for more robust detection.
        Accounts for star points to avoid false positives.
        
        Args:
            cell: Cell image to analyze
            board_x: X coordinate on the board (0-18)
            board_y: Y coordinate on the board (0-18)
        """
        if cell.size == 0 or self.stone_size is None:
            return StoneColor.EMPTY
            
        # Convert to grayscale if needed
        if len(cell.shape) == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell
            
        # Get cell dimensions
        height, width = gray.shape
        center_y, center_x = height // 2, width // 2
        
        # Create masks for different regions
        stone_mask = np.zeros_like(gray)
        stone_radius = int(self.stone_size * 0.4)
        cv2.circle(stone_mask, (center_x, center_y), stone_radius, 255, -1)
        
        # Create background mask (ring around the stone area)
        bg_mask = np.zeros_like(gray)
        outer_radius = int(self.stone_size * 0.6)
        inner_radius = int(self.stone_size * 0.45)
        cv2.circle(bg_mask, (center_x, center_y), outer_radius, 255, -1)
        cv2.circle(bg_mask, (center_x, center_y), inner_radius, 0, -1)
        
        # Create star point mask (smaller area)
        star_mask = np.zeros_like(gray)
        star_radius = int(self.stone_size * 0.15)
        cv2.circle(star_mask, (center_x, center_y), star_radius, 255, -1)
        
        # Apply masks and get statistics
        stone_area = cv2.bitwise_and(gray, gray, mask=stone_mask)
        bg_area = cv2.bitwise_and(gray, gray, mask=bg_mask)
        star_area = cv2.bitwise_and(gray, gray, mask=star_mask)
        
        stone_pixels = stone_area[stone_mask > 0]
        bg_pixels = bg_area[bg_mask > 0]
        star_pixels = star_area[star_mask > 0]
        
        if len(stone_pixels) == 0 or len(bg_pixels) == 0:
            return StoneColor.EMPTY
            
        # Calculate statistics
        stone_mean = np.mean(stone_pixels)
        stone_std = np.std(stone_pixels)
        bg_mean = np.mean(bg_pixels)
        bg_std = np.std(bg_pixels)
        
        # Calculate contrast
        contrast = abs(stone_mean - bg_mean)
        
        # Check if this is a star point position
        is_star_point = (board_x, board_y) in self.star_points
        
        if is_star_point:
            if len(star_pixels) > 0:
                star_mean = np.mean(star_pixels)
                star_std = np.std(star_pixels)
                star_contrast = abs(star_mean - bg_mean)
                
                # Star point characteristics
                if star_contrast < 30 and star_std < 15:
                    # This is likely just a star point
                    return StoneColor.EMPTY
        
        # Enhanced stone detection with multiple criteria
        is_black = False
        is_white = False
        
        # Contrast thresholds
        min_contrast = 45 if is_star_point else 35
        
        # Black stone criteria
        if stone_mean < bg_mean - min_contrast:
            if stone_std < 35:  # Black stones are usually more uniform
                # Verify shape using adaptive threshold
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                stone_shape = cv2.bitwise_and(thresh, stone_mask)
                stone_area_ratio = np.sum(stone_shape > 0) / np.sum(stone_mask > 0)
                
                if stone_area_ratio > 0.7:  # At least 70% filled
                    is_black = True
        
        # White stone criteria
        elif stone_mean > bg_mean + min_contrast:
            if stone_std < 45:  # Allow slightly higher variance for white stones
                # Verify shape using adaptive threshold
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                stone_shape = cv2.bitwise_and(thresh, stone_mask)
                stone_area_ratio = np.sum(stone_shape > 0) / np.sum(stone_mask > 0)
                
                if stone_area_ratio > 0.6:  # Slightly lower threshold for white stones
                    is_white = True
        
        # Additional verification for star point positions
        if is_star_point and (is_black or is_white):
            # Require stronger evidence at star points
            if contrast < min_contrast * 1.5:  # 50% higher contrast required
                return StoneColor.EMPTY
            
            # Check edge response
            edges = cv2.Canny(gray, 50, 150)
            edge_mask = cv2.bitwise_and(edges, stone_mask)
            edge_ratio = np.sum(edge_mask > 0) / (2 * np.pi * stone_radius)
            
            if edge_ratio < 0.6:  # Require strong edge response
                return StoneColor.EMPTY
        
        if is_black:
            return StoneColor.BLACK
        elif is_white:
            return StoneColor.WHITE
            
        return StoneColor.EMPTY
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in [top-left, top-right, bottom-right, bottom-left] order"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Convert points to 2D array if needed
        if len(pts.shape) == 3:
            pts = pts.reshape(pts.shape[0], 2)
        
        # Get points sum and diff
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        
        # Order points
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        rect[1] = pts[np.argmin(d)]  # Top-right
        rect[3] = pts[np.argmax(d)]  # Bottom-left
        
        return rect
    
    def _detect_grid_section(self, processed: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], List[np.ndarray], Tuple[int, int], str]]:
        """
        Detect grid lines and estimate which section of the board is visible
        Returns: (board_region, grid_lines, (start_x, start_y), corner_type)
        corner_type can be: 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'middle'
        """
        # Use probabilistic Hough transform for line detection
        edges = cv2.Canny(processed, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=80,
            minLineLength=30,
            maxLineGap=5
        )
        
        if lines is None or len(lines) < self.min_grid_lines:
            return None
        
        # Cluster lines to find grid
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length < 30:  # Skip short lines
                continue
                
            if angle < 45:  # Vertical lines
                vertical_lines.append(line[0])
            elif angle > 45:  # Horizontal lines
                horizontal_lines.append(line[0])
        
        if len(horizontal_lines) < self.min_grid_lines or len(vertical_lines) < self.min_grid_lines:
            return None
        
        # Sort lines by position
        horizontal_lines.sort(key=lambda l: l[1])  # Sort by y coordinate
        vertical_lines.sort(key=lambda l: l[0])    # Sort by x coordinate
        
        # Calculate average line spacing
        h_spacing = np.median([horizontal_lines[i+1][1] - horizontal_lines[i][1] 
                             for i in range(len(horizontal_lines)-1)])
        v_spacing = np.median([vertical_lines[i+1][0] - vertical_lines[i][0] 
                             for i in range(len(vertical_lines)-1)])
        
        # Get board region
        x_min = min(l[0] for l in vertical_lines)
        x_max = max(l[0] for l in vertical_lines)
        y_min = min(l[1] for l in horizontal_lines)
        y_max = max(l[1] for l in horizontal_lines)
        
        # Add margin to region
        margin = int(max(h_spacing, v_spacing) * 0.5)
        board_region = (
            max(0, x_min - margin),
            max(0, y_min - margin),
            min(processed.shape[1], x_max + margin),
            min(processed.shape[0], y_max + margin)
        )
        
        # Count visible lines
        visible_cols = len(vertical_lines)
        visible_rows = len(horizontal_lines)
        
        # Calculate approximate line spacing in full board
        full_board_spacing = min(
            (processed.shape[0] - 2 * self.board_margin) / (self.board_size - 1),
            (processed.shape[1] - 2 * self.board_margin) / (self.board_size - 1)
        )
        
        # Analyze line patterns to determine board section
        first_v_spacing = vertical_lines[1][0] - vertical_lines[0][0]
        last_v_spacing = vertical_lines[-1][0] - vertical_lines[-2][0]
        first_h_spacing = horizontal_lines[1][1] - horizontal_lines[0][1]
        last_h_spacing = horizontal_lines[-1][1] - horizontal_lines[-2][1]
        
        # Compare with median spacing to determine if we're at an edge
        v_median = np.median([vertical_lines[i+1][0] - vertical_lines[i][0] 
                            for i in range(len(vertical_lines)-1)])
        h_median = np.median([horizontal_lines[i+1][1] - horizontal_lines[i][1] 
                            for i in range(len(horizontal_lines)-1)])
        
        # Determine if we're at board edges
        at_left = abs(first_v_spacing - v_median) / v_median < 0.2
        at_right = abs(last_v_spacing - v_median) / v_median < 0.2
        at_top = abs(first_h_spacing - h_median) / h_median < 0.2
        at_bottom = abs(last_h_spacing - h_median) / h_median < 0.2
        
        # Determine corner type and starting position
        corner_type = 'middle'
        start_x = 0
        start_y = 0
        
        if at_left and at_top:
            corner_type = 'top_left'
            start_x = 0
            start_y = 0
        elif at_right and at_top:
            corner_type = 'top_right'
            start_x = self.board_size - visible_cols + 1
            start_y = 0
        elif at_left and at_bottom:
            corner_type = 'bottom_left'
            start_x = 0
            start_y = self.board_size - visible_rows + 1
        elif at_right and at_bottom:
            corner_type = 'bottom_right'
            start_x = self.board_size - visible_cols + 1
            start_y = self.board_size - visible_rows + 1
        else:
            # Estimate position based on visible star points
            star_points_visible = []
            for y in range(visible_rows):
                for x in range(visible_cols):
                    if (x, y) in self.star_points:
                        star_points_visible.append((x, y))
            
            if star_points_visible:
                # Use the first visible star point as reference
                ref_x, ref_y = star_points_visible[0]
                # Calculate offset to nearest star point
                for sx, sy in self.star_points:
                    if abs(sx - ref_x) <= visible_cols and abs(sy - ref_y) <= visible_rows:
                        start_x = max(0, sx - ref_x)
                        start_y = max(0, sy - ref_y)
                        break
        
        # Validate and adjust start positions
        start_x = min(max(0, start_x), self.board_size - visible_cols + 1)
        start_y = min(max(0, start_y), self.board_size - visible_rows + 1)
        
        logger.info(f"Detected board section: {corner_type} starting at ({start_x}, {start_y})")
        logger.info(f"Visible area: {visible_cols}x{visible_rows}")
        
        return board_region, (horizontal_lines, vertical_lines), (start_x, start_y), corner_type
