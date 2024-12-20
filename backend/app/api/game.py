from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional, Tuple
from pydantic import BaseModel
from ..models.go_board import GoBoard, StoneColor
from ..models.joseki import JosekiDatabase
from ..models.life_death import LifeDeathAnalyzer
from ..services.board_recognition import BoardRecognitionService
import os
import tempfile

class MoveRequest(BaseModel):
    x: int
    y: int
    color: str

router = APIRouter()
board_recognition = BoardRecognitionService()
game_boards = {}  # Store active game boards
joseki_db = JosekiDatabase()

@router.post("/games/new")
async def create_new_game(board_size: int = 19):
    """Create a new game"""
    game_id = len(game_boards)
    game_boards[game_id] = GoBoard(size=board_size)
    return {
        "game_id": game_id,
        "current_color": StoneColor.BLACK.value
    }

@router.post("/games/{game_id}/move")
async def make_move(game_id: int, move: MoveRequest):
    """Make a move on the board"""
    if game_id not in game_boards:
        raise HTTPException(status_code=404, detail="Game not found")
        
    board = game_boards[game_id]
    if not board.place_stone(move.x, move.y, move.color):
        raise HTTPException(status_code=400, detail="Invalid move")
        
    # Check for matching joseki patterns
    moves = board.get_move_history()
    matching_patterns = joseki_db.find_matching_pattern(moves)
    
    # Analyze the position after the move
    analyzer = LifeDeathAnalyzer(board)
    position_analysis = analyzer.analyze_group(move.x, move.y)
    
    return {
        "success": True,
        "joseki": [p.name for p in matching_patterns],
        "position_analysis": position_analysis,
        "can_undo": board.can_undo(),
        "can_redo": board.can_redo(),
        "current_color": board.get_current_color()
    }

@router.post("/games/{game_id}/undo")
async def undo_move(game_id: int):
    """Undo the last move"""
    if game_id not in game_boards:
        raise HTTPException(status_code=404, detail="Game not found")
        
    board = game_boards[game_id]
    last_move = board.undo_move()
    if last_move is None:
        raise HTTPException(status_code=400, detail="No moves to undo")
        
    return {
        "last_move": last_move,
        "can_undo": board.can_undo(),
        "can_redo": board.can_redo(),
        "current_color": board.get_current_color()
    }

@router.post("/games/{game_id}/redo")
async def redo_move(game_id: int):
    """Redo a previously undone move"""
    if game_id not in game_boards:
        raise HTTPException(status_code=404, detail="Game not found")
        
    board = game_boards[game_id]
    next_move = board.redo_move()
    if next_move is None:
        raise HTTPException(status_code=400, detail="No moves to redo")
        
    return {
        "next_move": next_move,
        "can_undo": board.can_undo(),
        "can_redo": board.can_redo(),
        "current_color": board.get_current_color()
    }

@router.get("/games/{game_id}/state")
async def get_game_state(game_id: int):
    """Get current game state"""
    if game_id not in game_boards:
        raise HTTPException(status_code=404, detail="Game not found")
        
    board = game_boards[game_id]
    return {
        "board": board.get_board_state(),
        "history": board.get_move_history(),
        "can_undo": board.can_undo(),
        "can_redo": board.can_redo(),
        "current_color": board.get_current_color()
    }

@router.get("/games/{game_id}/analysis")
async def analyze_position(game_id: int, x: int, y: int):
    """Analyze a position on the board"""
    if game_id not in game_boards:
        raise HTTPException(status_code=404, detail="Game not found")
        
    board = game_boards[game_id]
    if not board.is_valid_position(x, y):
        raise HTTPException(status_code=400, detail="Invalid position")
        
    analyzer = LifeDeathAnalyzer(board)
    analysis = analyzer.analyze_group(x, y)
    vital_points = analyzer.find_vital_points(x, y)
    
    return {
        "analysis": analysis,
        "vital_points": vital_points
    }

@router.get("/games/{game_id}/joseki")
async def get_joseki_suggestions(game_id: int):
    """Get joseki suggestions for the current position"""
    if game_id not in game_boards:
        raise HTTPException(status_code=404, detail="Game not found")
        
    board = game_boards[game_id]
    moves = board.get_move_history()
    
    matching_patterns = joseki_db.find_matching_pattern(moves)
    next_moves = joseki_db.get_next_moves(moves)
    
    return {
        "matching_patterns": [
            {"name": p.name, "description": p.description}
            for p in matching_patterns
        ],
        "suggested_moves": next_moves,
        "current_color": board.get_current_color()
    }

@router.post("/recognition/board")
async def recognize_board(image: UploadFile = File(...)):
    """Recognize Go board from uploaded image"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = await image.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Process the image
        board_state = board_recognition.detect_board(temp_path)
        if board_state is None:
            raise HTTPException(
                status_code=400, 
                detail="Could not detect board in image"
            )
        return {"board_state": board_state}
    finally:
        # Clean up temporary file
        os.unlink(temp_path)
