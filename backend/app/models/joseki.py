from typing import List, Dict, Tuple
from .go_board import StoneColor

class JosekiPattern:
    def __init__(self, name: str, moves: List[Tuple[int, int, StoneColor]], description: str):
        self.name = name
        self.moves = moves
        self.description = description

class JosekiDatabase:
    def __init__(self):
        self.patterns: List[JosekiPattern] = []
        self._initialize_patterns()

    def _initialize_patterns(self):
        # Basic corner patterns
        self.patterns.extend([
            JosekiPattern(
                "星位挂角",
                [
                    (3, 3, StoneColor.BLACK),  # 小目
                    (4, 5, StoneColor.WHITE),  # 挂角
                    (5, 3, StoneColor.BLACK),  # 应对
                ],
                "最基本的角部定式之一，黑棋小目，白棋挂角。这是一个进取的下法。"
            ),
            JosekiPattern(
                "小目低挂",
                [
                    (3, 3, StoneColor.BLACK),  # 小目
                    (5, 3, StoneColor.WHITE),  # 低挂
                    (4, 2, StoneColor.BLACK),  # 应对
                ],
                "小目低挂是一种稳健的下法，白棋意图获得实地。"
            ),
            JosekiPattern(
                "高目定式",
                [
                    (3, 4, StoneColor.BLACK),  # 高目
                    (5, 3, StoneColor.WHITE),  # 低挂
                    (4, 5, StoneColor.BLACK),  # 应对
                ],
                "高目更注重外势的发展，这个定式展示了如何应对白棋的低挂。"
            ),
        ])

    def find_matching_pattern(self, moves: List[Tuple[int, int, StoneColor]]) -> List[JosekiPattern]:
        """Find joseki patterns that match the given sequence of moves"""
        matching_patterns = []
        
        for pattern in self.patterns:
            # Check if the moves match the start of the pattern
            if len(moves) <= len(pattern.moves):
                matches = True
                for i, move in enumerate(moves):
                    if move != pattern.moves[i]:
                        matches = False
                        break
                if matches:
                    matching_patterns.append(pattern)
        
        return matching_patterns

    def get_next_moves(self, moves: List[Tuple[int, int, StoneColor]]) -> List[Tuple[int, int, StoneColor]]:
        """Get possible next moves based on known patterns"""
        next_moves = []
        matching_patterns = self.find_matching_pattern(moves)
        
        for pattern in matching_patterns:
            if len(moves) < len(pattern.moves):
                next_move = pattern.moves[len(moves)]
                if next_move not in next_moves:
                    next_moves.append(next_move)
        
        return next_moves
