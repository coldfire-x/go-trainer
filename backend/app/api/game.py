from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from ..models.go_board import GoBoard, StoneColor
from ..models.life_death import LifeDeathAnalyzer
from ..services.board_recognition import BoardRecognitionService
from ..services.ml_board_recognition import MLBoardRecognitionService
import os
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")  # Add prefix here
board = None  # Initialize as None
analyzer = None
board_recognition = BoardRecognitionService()
ml_board_recognition = MLBoardRecognitionService()


class MoveRequest(BaseModel):
    x: int
    y: int
    color: str


class GameState(BaseModel):
    board: List[List[str]]
    current_color: str
    can_undo: bool
    can_redo: bool
    analysis: Optional[Dict[str, Any]] = None


def get_current_game() -> tuple[GoBoard, LifeDeathAnalyzer]:
    """Get current game or raise error if no game exists"""
    global board, analyzer
    if board is None or analyzer is None:
        raise HTTPException(
            status_code=400, detail="No game exists. Please create a new game first."
        )
    return board, analyzer


@router.post("/game/new")
async def create_new_game() -> GameState:
    """Create a new game"""
    global board, analyzer
    board = GoBoard()
    analyzer = LifeDeathAnalyzer(board)

    return GameState(
        board=board.get_board_state(),
        current_color=board.get_current_color().value,
        can_undo=board.can_undo(),
        can_redo=board.can_redo(),
        analysis=None,
    )


@router.post("/move")
async def make_move(move: MoveRequest) -> GameState:
    board, analyzer = get_current_game()

    # Convert string color to StoneColor enum
    try:
        stone_color = StoneColor(move.color)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid color")

    # Validate coordinates
    if not (0 <= move.x < board.size and 0 <= move.y < board.size):
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    # Try to place the stone
    if not board.place_stone(move.x, move.y, stone_color):
        raise HTTPException(status_code=400, detail="Invalid move")

    # Analyze the group at the move position
    position_analysis = analyzer.analyze_group(move.x, move.y)

    # Return the new game state
    return GameState(
        board=board.get_board_state(),
        current_color=board.get_current_color().value,
        can_undo=board.can_undo(),
        can_redo=board.can_redo(),
        analysis=position_analysis,
    )


@router.post("/undo")
async def undo_move() -> GameState:
    board, analyzer = get_current_game()

    if not board.can_undo():
        raise HTTPException(status_code=400, detail="No moves to undo")

    position = board.undo_move()

    # Get analysis for the last move if it exists
    analysis = None
    if position:
        x, y = position
        analysis = analyzer.analyze_group(x, y)

    return GameState(
        board=board.get_board_state(),
        current_color=board.get_current_color().value,
        can_undo=board.can_undo(),
        can_redo=board.can_redo(),
        analysis=analysis,
    )


@router.post("/redo")
async def redo_move() -> GameState:
    board, analyzer = get_current_game()

    if not board.can_redo():
        raise HTTPException(status_code=400, detail="No moves to redo")

    position = board.redo_move()

    # Get analysis for the redone move
    analysis = None
    if position:
        x, y = position
        analysis = analyzer.analyze_group(x, y)

    return GameState(
        board=board.get_board_state(),
        current_color=board.get_current_color().value,
        can_undo=board.can_undo(),
        can_redo=board.can_redo(),
        analysis=analysis,
    )


@router.get("/state")
async def get_state() -> GameState:
    board, analyzer = get_current_game()

    # Get the last move position if it exists
    analysis = None
    history = board.get_move_history()
    if history:
        x, y, _ = history[-1]
        analysis = analyzer.analyze_group(x, y)

    return GameState(
        board=board.get_board_state(),
        current_color=board.get_current_color().value,
        can_undo=board.can_undo(),
        can_redo=board.can_redo(),
        analysis=analysis,
    )


@router.post("/game/from_image")
async def create_game_from_image(
    image: UploadFile = File(...),
    use_ml: bool = Form(True)
) -> Dict[str, Any]:
    """
    Create a new game from an uploaded Go board image
    """
    logger.info(f"Processing image upload, use_ml: {use_ml}")
    try:
        # Create a temporary file to store the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(image.file, temp_file)
            temp_path = temp_file.name

        try:
            # Detect board state from image using ML or traditional CV
            if use_ml:
                logger.info("Using ML for board detection")
                board_state = ml_board_recognition.detect_board(temp_path)
            else:
                logger.info("Using traditional CV for board detection")
                board_state = board_recognition.detect_board(temp_path)
                
            if board_state is None:
                logger.error("Failed to detect board state")
                raise HTTPException(
                    status_code=400,
                    detail="Failed to detect Go board in the image"
                )

            # Create new game with detected board state
            global board
            board = GoBoard()
            
            # Apply detected moves
            stones_placed = 0
            for y in range(len(board_state)):
                for x in range(len(board_state[y])):
                    if board_state[y][x] != StoneColor.EMPTY:
                        if board.place_stone(x, y, board_state[y][x]):
                            stones_placed += 1
                        else:
                            logger.warning(
                                f"Failed to place stone at ({x}, {y}) "
                                f"with color {board_state[y][x]}"
                            )

            logger.info(f"Successfully placed {stones_placed} stones")
            
            # Get current board state
            current_state = board.get_board_state()
            logger.info(f"Current board state: {current_state}")

            response_data = {
                "message": f"Game created from image successfully with {stones_placed} stones",
                "board": current_state,
                "current_color": board.get_current_color(),
                "can_undo": board.can_undo(),
                "can_redo": board.can_redo()
            }
            
            logger.info(f"Sending response: {response_data}")
            return response_data

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )
