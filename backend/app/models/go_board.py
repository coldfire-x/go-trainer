from enum import Enum
from typing import List, Tuple, Set, Optional, Dict
from pydantic import BaseModel
import copy

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
        self.move_history: List[Tuple[int, int, str]] = []
        self.state_history: List[List[List[StoneColor]]] = []
        self.future_states: List[List[List[StoneColor]]] = []
        self.current_color = StoneColor.BLACK  # Track current color
        self.save_state()

    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def place_stone(self, x: int, y: int, color: str) -> bool:
        if not self.is_valid_position(x, y):
            return False
        if self.board[y][x] != StoneColor.EMPTY:
            return False
        if StoneColor(color) != self.current_color:
            return False  # Wrong color for current turn
            
        # Clear future states when making a new move
        self.future_states = []
        
        # Save current state before making the move
        self.save_state()
        
        # Place the stone
        self.board[y][x] = StoneColor(color)
        self.move_history.append((x, y, color))
        
        # Switch current color
        self.current_color = StoneColor.WHITE if self.current_color == StoneColor.BLACK else StoneColor.BLACK
        
        return True

    def remove_stone(self, x: int, y: int) -> bool:
        if not self.is_valid_position(x, y):
            return False
        if self.board[y][x] == StoneColor.EMPTY:
            return False
            
        self.board[y][x] = StoneColor.EMPTY
        return True

    def get_stone(self, x: int, y: int) -> Optional[str]:
        if not self.is_valid_position(x, y):
            return None
        return self.board[y][x].value

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group(self, x: int, y: int) -> List[Tuple[int, int]]:
        if not self.is_valid_position(x, y) or self.board[y][x] == StoneColor.EMPTY:
            return []

        color = self.board[y][x]
        visited = set()
        group = []
        stack = [(x, y)]

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            group.append(current)
            cx, cy = current

            for nx, ny in self.get_neighbors(cx, cy):
                if self.board[ny][nx] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))

        return group

    def save_state(self):
        """Save the current board state"""
        self.state_history.append([row[:] for row in self.board])

    def restore_state(self, state: List[List[StoneColor]]):
        """Restore the board to a given state"""
        self.board = [row[:] for row in state]

    def undo_move(self) -> Optional[Tuple[int, int, str]]:
        """Undo the last move and return the undone move"""
        if not self.state_history:
            return None
            
        # Save current state to future states for redo
        self.future_states.append([row[:] for row in self.board])
        
        # Restore previous state
        previous_state = self.state_history.pop()
        self.restore_state(previous_state)
        
        if self.move_history:
            last_move = self.move_history.pop()
            # Switch back to previous color
            self.current_color = StoneColor(last_move[2])
            return last_move
            
        return None

    def redo_move(self) -> Optional[Tuple[int, int, str]]:
        """Redo a previously undone move"""
        if not self.future_states:
            return None
            
        # Save current state to history
        self.save_state()
        
        # Restore next state
        next_state = self.future_states.pop()
        self.restore_state(next_state)
        
        # Find the move that was redone by comparing states
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] != StoneColor.EMPTY and (x, y, self.board[y][x].value) not in self.move_history:
                    move = (x, y, self.board[y][x].value)
                    self.move_history.append(move)
                    # Switch to next color
                    self.current_color = StoneColor.WHITE if StoneColor(move[2]) == StoneColor.BLACK else StoneColor.BLACK
                    return move
                    
        return None

    def can_undo(self) -> bool:
        """Check if there are moves that can be undone"""
        return len(self.state_history) > 0

    def can_redo(self) -> bool:
        """Check if there are moves that can be redone"""
        return len(self.future_states) > 0

    def get_board_state(self) -> List[List[str]]:
        """Get the current board state"""
        return [[stone.value for stone in row] for row in self.board]

    def get_move_history(self) -> List[Tuple[int, int, str]]:
        """Get the move history"""
        return self.move_history[:]

    def get_current_color(self) -> str:
        """Get the current player's color"""
        return self.current_color.value
