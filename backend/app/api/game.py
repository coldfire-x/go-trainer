from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from ..models.go_board import GoBoard, StoneColor
from ..models.life_death import LifeDeathAnalyzer

router = APIRouter()
board = None  # Initialize as None
analyzer = None


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
