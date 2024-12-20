from typing import List, Set, Tuple, Optional
from .go_board import GoBoard, StoneColor

class LifeDeathAnalyzer:
    def __init__(self, board: GoBoard):
        self.board = board

    def _get_group(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """Get all stones in the same group"""
        color = self.board.board[y][x]
        if color == StoneColor.EMPTY:
            return set()

        group = set()
        to_check = {(x, y)}

        while to_check:
            current = to_check.pop()
            if current in group:
                continue

            group.add(current)
            cx, cy = current

            for nx, ny in self.board.get_neighbors(cx, cy):
                if self.board.board[ny][nx] == color and (nx, ny) not in group:
                    to_check.add((nx, ny))

        return group

    def _count_eyes(self, group: Set[Tuple[int, int]]) -> int:
        """Count the number of eyes in a group"""
        eyes = set()
        potential_eyes = set()

        # Find empty points surrounded by the group
        for x, y in group:
            for nx, ny in self.board.get_neighbors(x, y):
                if self.board.board[ny][nx] == StoneColor.EMPTY:
                    potential_eyes.add((nx, ny))

        # Check each potential eye
        for x, y in potential_eyes:
            surrounded = True
            diagonal_empty = 0
            
            # Check adjacent points
            for nx, ny in self.board.get_neighbors(x, y):
                if (nx, ny) not in group:
                    surrounded = False
                    break
            
            # Check diagonal points
            diagonals = [
                (x+1, y+1), (x+1, y-1),
                (x-1, y+1), (x-1, y-1)
            ]
            for dx, dy in diagonals:
                if (self.board.is_valid_position(dx, dy) and 
                    self.board.board[dy][dx] == StoneColor.EMPTY):
                    diagonal_empty += 1
            
            if surrounded and diagonal_empty <= 1:
                eyes.add((x, y))

        return len(eyes)

    def analyze_group(self, x: int, y: int) -> dict:
        """Analyze the life and death status of a group"""
        group = self._get_group(x, y)
        if not group:
            return {"status": "empty"}

        eyes = self._count_eyes(group)
        liberties = sum(self.board.count_liberties(x, y) for x, y in group)
        
        status = "unknown"
        if eyes >= 2:
            status = "alive"
        elif eyes == 0 and liberties <= 2:
            status = "in_danger"
        elif eyes == 1 and liberties <= 3:
            status = "in_danger"
        elif liberties >= 6:
            status = "likely_alive"

        return {
            "status": status,
            "group_size": len(group),
            "eyes": eyes,
            "liberties": liberties
        }

    def find_vital_points(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Find vital points for attacking or defending a group"""
        group = self._get_group(x, y)
        vital_points = []
        
        if not group:
            return vital_points

        # Check liberties
        for gx, gy in group:
            for nx, ny in self.board.get_neighbors(gx, gy):
                if self.board.board[ny][nx] == StoneColor.EMPTY:
                    # Simulate placing a stone
                    original_board = [row[:] for row in self.board.board]
                    opposite_color = (StoneColor.WHITE if self.board.board[gy][gx] == StoneColor.BLACK 
                                   else StoneColor.BLACK)
                    
                    self.board.board[ny][nx] = opposite_color
                    if self.board.count_liberties(gx, gy) == 0:
                        vital_points.append((nx, ny))
                    
                    # Restore board
                    self.board.board = original_board

        return vital_points
