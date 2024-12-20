from typing import List, Set, Tuple, Optional
from .go_board import StoneColor

class LifeDeathAnalyzer:
    def __init__(self, board):
        self.board = board

    def _get_group(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """Get all stones in the same group"""
        if not self.board.is_valid_position(x, y):
            return set()
            
        color = self.board.get_stone(x, y)
        if color == StoneColor.EMPTY:
            return set()
            
        group = set()
        to_check = [(x, y)]
        
        while to_check:
            cx, cy = to_check.pop()
            if (cx, cy) in group:
                continue
                
            group.add((cx, cy))
            for nx, ny in self.board.get_neighbors(cx, cy):
                if (nx, ny) not in group and self.board.get_stone(nx, ny) == color:
                    to_check.append((nx, ny))
                    
        return group

    def _get_liberties(self, group: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Get all liberties of a group"""
        liberties = set()
        for x, y in group:
            for nx, ny in self.board.get_neighbors(x, y):
                if self.board.get_stone(nx, ny) == StoneColor.EMPTY:
                    liberties.add((nx, ny))
        return liberties

    def _get_adjacent_groups(self, group: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
        """Get all adjacent groups of opposite color"""
        if not group:
            return []
            
        # Get color of the group
        x, y = next(iter(group))
        color = self.board.get_stone(x, y)
        
        # Find all adjacent points
        adjacent_points = set()
        for x, y in group:
            for nx, ny in self.board.get_neighbors(x, y):
                stone = self.board.get_stone(nx, ny)
                if stone != StoneColor.EMPTY and stone != color:
                    adjacent_points.add((nx, ny))
                    
        # Get groups for each adjacent point
        adjacent_groups = []
        processed_points = set()
        
        for x, y in adjacent_points:
            if (x, y) not in processed_points:
                adj_group = self._get_group(x, y)
                adjacent_groups.append(adj_group)
                processed_points.update(adj_group)
                
        return adjacent_groups

    def analyze_group(self, x: int, y: int) -> dict:
        """Analyze the life and death status of a group"""
        group = self._get_group(x, y)
        if not group:
            return {"status": "empty"}
            
        liberties = self._get_liberties(group)
        adjacent_groups = self._get_adjacent_groups(group)
        
        # Get color of the group
        color = self.board.get_stone(x, y)
        
        analysis = {
            "color": color.value,
            "size": len(group),
            "liberties": len(liberties),
            "liberty_points": list(liberties),
            "adjacent_groups": len(adjacent_groups),
            "group_points": list(group)
        }
        
        # Basic life and death analysis
        if len(liberties) >= 4:
            analysis["status"] = "alive"
            analysis["reason"] = "Group has 4 or more liberties"
        elif len(liberties) <= 1:
            analysis["status"] = "in_danger"
            analysis["reason"] = "Group has only 1 liberty"
        else:
            # Check for eye formation potential
            eye_potential = self._analyze_eye_potential(group)
            if eye_potential >= 2:
                analysis["status"] = "alive"
                analysis["reason"] = f"Group can form {eye_potential} eyes"
            elif eye_potential == 1:
                analysis["status"] = "uncertain"
                analysis["reason"] = "Group can form 1 eye"
            else:
                analysis["status"] = "in_danger"
                analysis["reason"] = "Group cannot form eyes"
            
        return analysis

    def _analyze_eye_potential(self, group: Set[Tuple[int, int]]) -> int:
        """Analyze the potential for eye formation"""
        potential_eyes = 0
        checked_points = set()
        
        # Check each empty point adjacent to the group
        for x, y in group:
            for nx, ny in self.board.get_neighbors(x, y):
                if (nx, ny) in checked_points:
                    continue
                    
                if self.board.get_stone(nx, ny) == StoneColor.EMPTY:
                    checked_points.add((nx, ny))
                    if self._is_potential_eye(nx, ny, group):
                        potential_eyes += 1
                        
        return potential_eyes

    def _is_potential_eye(self, x: int, y: int, group: Set[Tuple[int, int]]) -> bool:
        """Check if a point can potentially become an eye"""
        # Count how many adjacent points are part of the group
        group_adjacent = 0
        for nx, ny in self.board.get_neighbors(x, y):
            if (nx, ny) in group:
                group_adjacent += 1
                
        # For a potential eye, most adjacent points should be part of the group
        return group_adjacent >= 3

    def find_vital_points(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Find vital points for attack or defense of a group"""
        group = self._get_group(x, y)
        if not group:
            return []
            
        liberties = self._get_liberties(group)
        vital_points = []
        
        # For groups with few liberties, all liberty points are vital
        if len(liberties) <= 2:
            vital_points.extend(liberties)
            
        # Add points that would connect to other friendly groups
        color = self.board.get_stone(x, y)
        for lx, ly in liberties:
            for nx, ny in self.board.get_neighbors(lx, ly):
                if self.board.get_stone(nx, ny) == color and (nx, ny) not in group:
                    vital_points.append((lx, ly))
                    break
                    
        return list(set(vital_points))  # Remove duplicates
