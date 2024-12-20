from enum import Enum
from typing import List, Tuple, Set, Optional
from pydantic import BaseModel


class StoneColor(str, Enum):
    BLACK = "black"
    WHITE = "white"
    EMPTY = "empty"


class Stone(BaseModel):
    x: int
    y: int
    color: StoneColor


class BoardState(BaseModel):
    board: List[List[StoneColor]]
    move_history: List[Stone]


class GoBoard:
    def __init__(self, size: int = 19):
        self.size = size
        self.board = [[StoneColor.EMPTY for _ in range(size)] for _ in range(size)]
        self.move_history: List[Tuple[int, int, StoneColor]] = []  # (x, y, color)
        self.undone_moves: List[
            Tuple[int, int, StoneColor]
        ] = []  # Store undone moves for redo
        self.current_color = StoneColor.BLACK  # Track current color

    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if a position is within the board boundaries"""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_stone(self, x: int, y: int) -> StoneColor:
        """Get the stone at a position"""
        if not self.is_valid_position(x, y):
            return StoneColor.EMPTY
        return self.board[y][x]

    def place_stone(self, x: int, y: int, color: StoneColor) -> bool:
        """Place a stone on the board"""
        if not self.is_valid_position(x, y):
            return False

        if self.board[y][x] != StoneColor.EMPTY:
            return False

        if color != self.current_color:
            return False

        # Place the stone
        self.board[y][x] = color

        # Add to history and clear undone moves
        self.move_history.append((x, y, color))
        self.undone_moves.clear()

        # Switch current color
        self.current_color = (
            StoneColor.WHITE if color == StoneColor.BLACK else StoneColor.BLACK
        )

        return True

    def remove_stone(self, x: int, y: int) -> bool:
        """Remove a stone from the board"""
        if not self.is_valid_position(x, y):
            return False
        if self.board[y][x] == StoneColor.EMPTY:
            return False

        self.board[y][x] = StoneColor.EMPTY
        return True

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """Get all stones in the same group (connected stones of the same color)"""
        if not self.is_valid_position(x, y):
            return set()

        stone = self.get_stone(x, y)
        if stone == StoneColor.EMPTY:
            return set()

        group = set()
        to_check = [(x, y)]

        while to_check:
            cx, cy = to_check.pop()
            if (cx, cy) in group:
                continue

            group.add((cx, cy))
            for nx, ny in self.get_neighbors(cx, cy):
                if self.get_stone(nx, ny) == stone and (nx, ny) not in group:
                    to_check.append((nx, ny))

        return group

    def get_liberties(self, group: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Get all liberties of a group of stones"""
        liberties = set()
        for x, y in group:
            for nx, ny in self.get_neighbors(x, y):
                if self.get_stone(nx, ny) == StoneColor.EMPTY:
                    liberties.add((nx, ny))
        return liberties

    def undo_move(self) -> Optional[Tuple[int, int]]:
        """Undo the last move"""
        if not self.move_history:
            return None

        x, y, color = self.move_history.pop()
        self.undone_moves.append((x, y, color))
        self.board[y][x] = StoneColor.EMPTY

        # Switch current color back
        self.current_color = color

        return (x, y)

    def redo_move(self) -> Optional[Tuple[int, int]]:
        """Redo a previously undone move"""
        if not self.undone_moves:
            return None

        x, y, color = self.undone_moves.pop()
        self.move_history.append((x, y, color))
        self.board[y][x] = color

        # Switch current color
        self.current_color = (
            StoneColor.WHITE if color == StoneColor.BLACK else StoneColor.BLACK
        )

        return (x, y)

    def can_undo(self) -> bool:
        """Check if there are moves that can be undone"""
        return len(self.move_history) > 0

    def can_redo(self) -> bool:
        """Check if there are moves that can be redone"""
        return len(self.undone_moves) > 0

    def get_board_state(self) -> List[List[str]]:
        """Get the current board state"""
        return [[stone.value for stone in row] for row in self.board]

    def get_move_history(self) -> List[Tuple[int, int, StoneColor]]:
        """Get the history of moves"""
        return self.move_history.copy()

    def get_current_color(self) -> StoneColor:
        """Get the current player's color"""
        return self.current_color
