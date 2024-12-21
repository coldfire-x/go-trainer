import cv2
import numpy as np
from app.services.board_recognition import BoardRecognitionService
import os
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_detection(image_path: str, board_service: BoardRecognitionService):
    """Visualize the board detection process"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return False
        
        # Get original dimensions
        orig_height, orig_width = img.shape[:2]
        
        # Resize if image is too large
        max_size = 1200
        if max(orig_height, orig_width) > max_size:
            scale = max_size / max(orig_height, orig_width)
            img = cv2.resize(img, (int(orig_width * scale), int(orig_height * scale)))
            logger.info(f"Resized image to {img.shape[1]}x{img.shape[0]}")
        
        # Preprocess image
        processed = board_service._preprocess_image(img)
        
        # Detect grid section
        grid_info = board_service._detect_grid_section(processed)
        if grid_info is None:
            logger.error("Failed to detect grid section")
            return False
        
        board_region, (horizontal_lines, vertical_lines), (start_x, start_y), corner_type = grid_info
        
        # Draw detected grid section
        grid_viz = img.copy()
        
        # Draw board region
        cv2.rectangle(grid_viz, 
                     (board_region[0], board_region[1]),
                     (board_region[2], board_region[3]),
                     (0, 255, 0), 2)
        
        # Add text showing board section and position
        cv2.putText(grid_viz, 
                   f"Board Section: {corner_type}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(grid_viz,
                   f"Starting at: ({start_x}, {start_y})",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw grid lines
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(grid_viz, (x1, y1), (x2, y2), (255, 0, 0), 1)
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(grid_viz, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        # Extract and transform board region
        board_img = img[board_region[1]:board_region[3], 
                       board_region[0]:board_region[2]]
        warped = board_service._get_board_transform(
            processed[board_region[1]:board_region[3], 
                     board_region[0]:board_region[2]],
            board_img
        )
        if warped is None:
            logger.error("Failed to transform board region")
            return False
        
        # Create visualization of cell analysis
        cell_viz = warped.copy()
        height, width = warped.shape[:2]
        cell_width = (width - 2 * board_service.board_margin) / (board_service.board_size - 1)
        cell_height = (height - 2 * board_service.board_margin) / (board_service.board_size - 1)
        
        # Draw grid lines
        for i in range(board_service.board_size):
            x = int(board_service.board_margin + i * cell_width)
            y = int(board_service.board_margin + i * cell_height)
            cv2.line(cell_viz, (board_service.board_margin, y), 
                    (width - board_service.board_margin, y), (0, 255, 0), 1)
            cv2.line(cell_viz, (x, board_service.board_margin), 
                    (x, height - board_service.board_margin), (0, 255, 0), 1)
        
        # Get cells and analyze stones
        cells = board_service._create_grid_cells(warped)
        stone_radius = board_service.stone_size // 2 if board_service.stone_size else int(cell_width * 0.4)
        
        # Draw detected stones and their positions
        visible_width = min(len(cells[0]), board_service.board_size - start_x)
        visible_height = min(len(cells), board_service.board_size - start_y)
        
        for y in range(visible_height):
            for x in range(visible_width):
                center_x = int(board_service.board_margin + x * cell_width)
                center_y = int(board_service.board_margin + y * cell_height)
                
                board_x = start_x + x
                board_y = start_y + y
                
                stone_color = board_service._analyze_cell(cells[y][x], board_x, board_y)
                if stone_color.value == 'black':
                    # Draw filled black circle
                    cv2.circle(cell_viz, (center_x, center_y), stone_radius, (0, 0, 0), -1)
                    # Draw detection circle
                    cv2.circle(cell_viz, (center_x, center_y), stone_radius + 2, (0, 255, 0), 2)
                    # Draw board position
                    cv2.putText(cell_viz, f"({board_x},{board_y})", 
                              (center_x-20, center_y-stone_radius-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                elif stone_color.value == 'white':
                    # Draw filled white circle
                    cv2.circle(cell_viz, (center_x, center_y), stone_radius, (255, 255, 255), -1)
                    # Draw black outline for white stones
                    cv2.circle(cell_viz, (center_x, center_y), stone_radius, (0, 0, 0), 1)
                    # Draw detection circle
                    cv2.circle(cell_viz, (center_x, center_y), stone_radius + 2, (0, 255, 0), 2)
                    # Draw board position
                    cv2.putText(cell_viz, f"({board_x},{board_y})", 
                              (center_x-20, center_y-stone_radius-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Draw star points in red
                if (board_x, board_y) in board_service.star_points:
                    star_radius = int(stone_radius * 0.3)
                    cv2.circle(cell_viz, (center_x, center_y), star_radius, 
                             (0, 0, 255) if stone_color == StoneColor.EMPTY else (255, 0, 0), -1)
        
        # Add text showing board section info
        cv2.putText(cell_viz, 
                   f"Board Section: {corner_type}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(cell_viz,
                   f"Visible Area: {visible_width}x{visible_height}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display results in windows
        cv2.namedWindow('Original with Grid', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Preprocessed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Warped with Stones', cv2.WINDOW_NORMAL)
        
        cv2.imshow('Original with Grid', grid_viz)
        cv2.imshow('Preprocessed', processed)
        cv2.imshow('Warped with Stones', cell_viz)
        
        # Save results
        output_dir = "test_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_grid.jpg"), grid_viz)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_processed.jpg"), processed)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_detected.jpg"), cell_viz)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return False

def main():
    try:
        # Initialize board recognition service
        board_service = BoardRecognitionService()
        
        # Test directory
        test_dir = "test_images"
        if not os.path.exists(test_dir):
            logger.error(f"Test directory {test_dir} does not exist")
            return
        
        # Process each image in test directory
        image_files = [f for f in os.listdir(test_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            logger.error(f"No image files found in {test_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for filename in image_files:
            if filename.startswith('.'):  # Skip hidden files
                continue
                
            image_path = os.path.join(test_dir, filename)
            logger.info(f"\nProcessing {filename}...")
            
            # Visualize detection
            if visualize_detection(image_path, board_service):
                # Get board state and corner type
                board_state, corner_type = board_service.detect_board(image_path)
                if board_state:
                    logger.info(f"Board section detected: {corner_type}")
                    logger.info("Board state detected:")
                    for row in board_state:
                        print(" ".join(stone.value for stone in row))
                else:
                    logger.error("Failed to detect board state")
            
            # Wait for key press
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Break if 'q' is pressed
            if key == ord('q'):
                break
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
