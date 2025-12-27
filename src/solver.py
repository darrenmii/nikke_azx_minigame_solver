import copy

def get_all_valid_moves(state, rows, cols):
    moves = []
    
    # Pre-calculate Integal Image (2D Prefix Sums)
    # P[r][c] stores sum of rectangle from (0,0) to (r,c)
    # Dimensions: rows x cols
    P = [[0] * cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            val = state[r][c]
            # S(r,c) = val + S(r-1, c) + S(r, c-1) - S(r-1, c-1)
            top = P[r-1][c] if r > 0 else 0
            left = P[r][c-1] if c > 0 else 0
            top_left = P[r-1][c-1] if (r > 0 and c > 0) else 0
            
            P[r][c] = val + top + left - top_left

    # Function to get sum of rectangle (r1, c1) to (r2, c2) in O(1)
    def get_rect_sum(r1, c1, r2, c2):
        # Result = P[r2][c2] - P[r1-1][c2] - P[r2][c1-1] + P[r1-1][c1-1]
        res = P[r2][c2]
        if r1 > 0:
            res -= P[r1-1][c2]
        if c1 > 0:
            res -= P[r2][c1-1]
        if r1 > 0 and c1 > 0:
            res += P[r1-1][c1-1]
        return res

    # Iterate all possible rectangles
    for r1 in range(rows):
        for c1 in range(cols):
            # Optimization: If P[rows-1][cols-1] (total sum) < 10, no rect can sum to 10? 
            # Not necessarily true locally, but globally yes. 
            # But inside loop, no easy early exit without more complex data structures.
            
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    s = get_rect_sum(r1, c1, r2, c2)
                    
                    if s == 10:
                        # Valid sum. Now ensure it's not empty (all zeros).
                        # Since all numbers >= 0, a sum of 10 implies at least one non-zero.
                        # So we don't need to iterate cells to check for non-zero!
                        # We just need to check s == 10.
                        moves.append((r1, c1, r2, c2))
                    elif s > 10:
                        # Since values are non-negative, expanding this rectangle to the right (increasing c2)
                        # will only increase or stay same.
                        # If current row expanding makes it > 10, further c2's in this row will also be > 10
                        # (assuming no negative numbers, which is true).
                        # So we can break the inner loop (c2).
                        break
    return moves

class Solver:
    def __init__(self):
        self.BEAM_WIDTH = 200 # Number of states to keep per depth

    def solve(self, matrix, progress_callback=None):
        """
        Finds the sequence of moves that eliminates the maximum number of items using Weighted Beam Search.
        """
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        
        # Initial State
        state_grid = []
        for r in range(rows):
            row_vals = []
            for c in range(cols):
                cell = matrix[r][c]
                val = cell['val'] if cell else 0
                row_vals.append(val)
            state_grid.append(tuple(row_vals))
            
        initial_state = tuple(state_grid)
        
        # Beam: List of (real_score, path, state, cum_weighted_score)
        # We start with 0 scores.
        current_beam = [(0, [], initial_state, 0)]
        
        best_overall_score = 0
        best_overall_path = []
        
        depth = 0
        while True:
            depth += 1
            if progress_callback:
                progress_callback(f"Depth {depth} - Beam Size {len(current_beam)}")
            
            next_beam_candidates = []
            no_moves_for_any = True
            
            # Expand every state in current beam
            for score, path, state, w_score in current_beam:
                moves = get_all_valid_moves(state, rows, cols)
                
                if not moves:
                    # Finalize this path
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_path = path
                    continue
                
                no_moves_for_any = False
                
                for move in moves:
                    r1, c1, r2, c2 = move
                    
                    # Apply move
                    move_eliminated = 0
                    large_items_eliminated = 0
                    new_state_list = [list(row) for row in state]
                    
                    for r in range(r1, r2 + 1):
                        for c in range(c1, c2 + 1):
                            val = state[r][c]
                            if val != 0:
                                move_eliminated += 1
                                if val >= 5: # HEURISTIC: Large items
                                    large_items_eliminated += 1
                            new_state_list[r][c] = 0
                    new_state = tuple(tuple(row) for row in new_state_list)
                    
                    new_score = score + move_eliminated
                    new_path = path + [move]
                    
                    # Bonus Weight Calculation
                    # +4 per large item eliminated.
                    # +10 if move size <= 2 (Pair Preference).
                    move_weight = move_eliminated + (large_items_eliminated * 4)
                    if move_eliminated <= 2:
                        move_weight += 10
                    
                    new_w_score = w_score + move_weight
                    
                    next_beam_candidates.append((new_score, new_path, new_state, new_w_score))
            
            if no_moves_for_any:
                break
            
            if not next_beam_candidates:
                break

            # Consolidate duplicates
            # If two paths lead to the same state, which one do we keep?
            # 1. Higher Real Score? (Objective truth)
            # 2. Higher Weighted Score? (Heuristic truth)
            # Generally, if the state is identical, the "better" path is the one that eliminated more items to get there.
            # Weighted score is just a bias for *future* potential. 
            # If state is same, future potential is same!
            # So we keep HIGHER REAL SCORE.
            
            unique_candidates = {} # State -> (real_score, path, w_score)
            for r_s, p, st, w_s in next_beam_candidates:
                if st not in unique_candidates:
                    unique_candidates[st] = (r_s, p, w_s)
                else:
                    # Tie-breaking
                    curr_rs, _, curr_ws = unique_candidates[st]
                    if r_s > curr_rs:
                        unique_candidates[st] = (r_s, p, w_s)
                    elif r_s == curr_rs and w_s > curr_ws:
                        # If real score equal, prefer one with higher heuristic?
                        unique_candidates[st] = (r_s, p, w_s)
            
            # Flatten
            candidates_list = [(r_s, p, st, w_s) for st, (r_s, p, w_s) in unique_candidates.items()]
            
            # Sort by WEIGHTED Score Descending for Beam Selection
            candidates_list.sort(key=lambda x: x[3], reverse=True)
            
            # Keep top K
            current_beam = candidates_list[:self.BEAM_WIDTH]
            
            # Track best result
            # Use sorted order? No, best result tracks REAL score.
            # candidate[0] is real score.
            
            # Find best real score in current beam for tracking
            current_best_real = max(candidates_list, key=lambda x: x[0])
            if current_best_real[0] > best_overall_score:
                best_overall_score = current_best_real[0]
                best_overall_path = current_best_real[1]

        return best_overall_path
