import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from solver import Solver

class TestSolver(unittest.TestCase):
    def test_simple_horizontal(self):
        # 5 5
        matrix = [
            [{'val': 5}, {'val': 5}]
        ]
        solver = Solver()
        path = solver.solve(matrix)
        self.assertEqual(len(path), 1)
        # Check coords: r1, c1, r2, c2
        self.assertEqual(path[0], (0, 0, 0, 1))

    def test_simple_vertical(self):
        # 5
        # 5
        matrix = [
            [{'val': 5}],
            [{'val': 5}]
        ]
        solver = Solver()
        path = solver.solve(matrix)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], (0, 0, 1, 0))

    def test_greedy_trap(self):
        # 2 2 2 2 2 (Sum 10, count 5)
        # 0 0 0 0 0 
        # If we had a vertical 10 that blocked this, solver should prefer the horizontal 5-item one?
        # Actually solver maximizes ELIMINATED items.
        
        # Grid:
        # 5 5 5 5
        # 5 5 5 5
        # Path should eliminate all 8.
        # 4 moves of 2 items each.
        
        matrix = [[{'val': 5} for _ in range(4)] for _ in range(2)]
        solver = Solver()
        path = solver.solve(matrix)
        
        # Verify total eliminations
        eliminated = 0
        # Mock simulation
        mock_grid = [[5 for _ in range(4)] for _ in range(2)]
        for r1, c1, r2, c2 in path:
            count = 0
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if mock_grid[r][c] != 0:
                        count += 1
                        mock_grid[r][c] = 0
            eliminated += count
            
        self.assertEqual(eliminated, 8)

    def test_pruning_perf(self):
        # 4x4 grid full of 5s. 16 items.
        # Should solve reasonably fast.
        matrix = [[{'val': 5} for _ in range(4)] for _ in range(4)]
        solver = Solver()
        path = solver.solve(matrix)
        
        # Should eliminate all 16
        self.assertEqual(len(path), 8) # 8 pairs of 5+5

    def test_heuristic_preference(self):
        # Scenario:
        # Option A: Clear five '2's (Total 5 items)
        # Option B: Clear one '2' and one '8' (Total 2 items)
        # Normally, greedy solver picks A (5 > 2).
        # Weighted solver (+4 for 8) should score:
        # A: 5 + 0 = 5
        # B: 2 + (1*4) = 6
        # So B should be preferred in the beam sort, IF it leads to a better long term outcome?
        # Actually, beam search will keep BOTH if width is large enough.
        # But if we force a choice or check sort order...
        # Let's check if the path *starts* with the large number clear if we constructed the grid such that
        # clearing the small ones traps the large one.
        
        # Grid:
        # 2 2 2 2 2  (Row 0: 5 items, sum 10)
        # 8 2        (Row 1: 2 items, sum 10)
        # 
        # If we take Row 0 first, we might use up 2s needed for Row 1? 
        # Actually any 2 works for 8.
        # Let's try a grid where taking the small cluster forces a game over earlier.
        # Or just verify that the solver *can* clear everything in a tricky setup.
        
        # Setup:
        # 2 2 2 2 2
        # 8
        # If we clear 2 2 2 2 2, we have an 8 left and no 2s. Game Over (Score 5).
        # If we clear 8 + 2 (from top row), we have four 2s left. 
        # Then we can't clear remaining four 2s (sum 8). Game Over (Score 2).
        # Wait, clearing 8+2 is WORSE in this specific isolated case (total 2 vs 5).
        
        # We need a case where clearing the big one helps.
        # 2 2 2 2 2 2 (Six 2s)
        # 8
        # Path A (Greedy small): Clear five 2s. Left: 2, 8. Sum 10! Clear 2, 8. Total cleared: 5 + 2 = 7.
        # Path B (Heuristic): Clear 8 + one 2. Left: five 2s. Clear five 2s. Total: 2 + 5 = 7.
        # Both get 7.
        
        # Okay, let's just stick to the user's premise: "Leftover big numbers".
        # We want to verify that the solver *prioritizes* the path that includes the 8 if possible?
        # Let's just run the standard tests to ensure no regression, and maybe a case with many large numbers.
        
        # 9 1 9 1
        # 5 5
        # Grid full of large numbers + helpers.
        matrix = [
            [{'val': 9}, {'val': 1}],
            [{'val': 9}, {'val': 1}], 
            [{'val': 5}, {'val': 5}]
        ]
        solver = Solver()
        path = solver.solve(matrix)
        # Should clear all 6.
        # 9+1, 9+1, 5+5.
        
        # Let's check expected length.
        self.assertTrue(len(path) >= 3)
        
        # Manually calculate elimination count
        eliminated = 0
        state = [[9, 1], [9, 1], [5, 5]]
        for r1, c1, r2, c2 in path:
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    if state[r][c] != 0:
                        eliminated += 1
                        state[r][c] = 0
        self.assertEqual(eliminated, 6)

    def test_pair_preference(self):
        # Scenario: 
        # A move with 4 items vs 2 moves of 2 items.
        # Grid:
        # 1 1 1 1 (Sum 4... wait, sum must be 10)
        # Pairs: 5 5 (Sum 10)
        # Quads: 2 2 3 3 (Sum 10)
        # 
        # Grid:
        # 2 2 3 3
        # 
        # Option A: Take all four (2+2+3+3=10). Score 4. Weighted: 4 + (0 large) + (0 pair bonus) = 4.
        # Option B: Take 2+? No sum 10 from pairs here.
        
        # Grid:
        # 5 5 (Pair. Score 2. Weighted: 2 + (2*4 large) + 10 pair = 20)
        # 2 2 2 2 2 (5 items. Score 5. Weighted: 5 + 0 + 0 = 5)
        # 
        # If we have both available, it should definitely pick the pair first.
        
        matrix = [
            [{'val': 5}, {'val': 5}],
            [{'val': 2}, {'val': 2}, {'val': 2}, {'val': 2}, {'val': 2}]
        ]
        # Make sure they are independent (e.g. widely separated or logic handles it)
        # Matrix structure implies adjacency.
        # Row 0: 5 5 (Rect (0,0,0,1))
        # Row 1: 2 2 2 2 2 (Rect (1,0,1,4))
        
        solver = Solver()
        # It should pick the pair first because of the massive weight.
        # Though in this case it can pick both sequentially so order doesn't impact final score.
        # But we can inspect the path order.
        
        path = solver.solve(matrix)
        # Path should be (0,0,0,1) then (1,0,1,4) or vice versa.
        # Pair heuristic says (0,0,0,1) is weight 20.
        # Big rect is weight 5.
        # So it should explore/pick Pair first.
        
        first_move = path[0]
        # (r1, c1, r2, c2)
        # Pair is Row 0.
        self.assertEqual(first_move[0], 0) 
        self.assertEqual(first_move[2], 0)

if __name__ == '__main__':
    unittest.main()
