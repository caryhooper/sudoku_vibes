# sudoku_vibes
# ===========
# A sudoku solver and puzzle collection using constraint propagation + backtracking.
#
# Vibe-coded by Big Pickle <3
# https://github.com/anomalyco/opencode
#
# Puzzles sourced from:
#   - Mt. Sudoku (https://api.mtsudoku.com) — easy, medium, hard, expert, master, extreme
#   - zlee1/SudokuSolver (GitHub) — generated boards
#   - Arto Inkala's "world's hardest sudoku" (via theguardian.com)
#
# License: MIT — do whatever you want, attribution appreciated.


from typing import Optional

Board = list[list[int]]
Candidates = list[list[set[int]]]

DIGITS: set[int] = set(range(1, 10))


def solve_sudoku(board: Board) -> Optional[Board]:
    """Solve a 9x9 sudoku board in-place using constraint propagation then backtracking.

    Phase 1 — constraint propagation: build a candidates grid and repeatedly
    apply naked-single and hidden-single rules until no more cells can be set.

    Phase 2 — backtracking: if the puzzle isn't fully solved after propagation,
    guess a value for the cell with the fewest remaining candidates (MRV heuristic)
    and recurse.  If a guess leads to a contradiction, undo and try the next value.

    Args:
        board: 9x9 grid where 0 represents an empty cell.

    Returns:
        The solved board (same list object, mutated in place), or None if
        the puzzle has no solution.
    """
    candidates: Candidates = get_candidates(board)
    if not constrain(board, candidates):
        return None
    if is_solved(board):
        return board
    return backtrack(board, candidates)


def get_candidates(board: Board) -> Candidates:
    """Build a 9x9 grid of candidate sets from the current board state.

    For each empty cell, the candidate set is {1..9} minus all digits already
    present in the same row, column, or 3x3 box.  Filled cells get an empty set.

    Args:
        board: Current board state.

    Returns:
        A 9x9 list-of-lists where candidates[r][c] is a set[int] of possible
        digits for cell (r, c).
    """
    candidates: Candidates = []
    for i in range(9):
        row_candidates: list[set[int]] = []
        for j in range(9):
            if board[i][j] != 0:
                row_candidates.append(set())
            else:
                used: set[int] = set()
                for k in range(9):
                    if board[i][k] != 0:
                        used.add(board[i][k])
                    if board[k][j] != 0:
                        used.add(board[k][j])
                start_row: int = i - i % 3
                start_col: int = j - j % 3
                for r in range(start_row, start_row + 3):
                    for c in range(start_col, start_col + 3):
                        if board[r][c] != 0:
                            used.add(board[r][c])
                row_candidates.append(DIGITS - used)
        candidates.append(row_candidates)
    return candidates


def constrain(board: Board, candidates: Candidates) -> bool:
    """Repeatedly apply naked-single and hidden-single rules until quiescence.

    Naked single: if a cell has exactly one candidate, set it.
    Hidden single: if a digit can only go in one cell of a row, column, or box,
    set it.

    Each time a cell is set, the candidates of affected peer cells are updated
    via eliminate().  If any cell ends up with zero candidates the puzzle is
    unsolvable and the function returns False.

    Args:
        board:    Board being solved (mutated in place).
        candidates: Candidate grid (mutated in place).

    Returns:
        True if propagation completed without contradiction, False otherwise.
    """
    changed: bool = True
    while changed:
        changed = False

        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    if len(candidates[i][j]) == 0:
                        return False
                    if len(candidates[i][j]) == 1:
                        val: int = next(iter(candidates[i][j]))
                        board[i][j] = val
                        candidates[i][j] = set()
                        changed = True
                        if not eliminate(candidates, i, j, val):
                            return False

        for i in range(9):
            for num in range(1, 10):
                cells_in_row: list[int] = [j for j in range(9) if board[i][j] == 0 and num in candidates[i][j]]
                if len(cells_in_row) == 1:
                    j: int = cells_in_row[0]
                    board[i][j] = num
                    candidates[i][j] = set()
                    changed = True
                    if not eliminate(candidates, i, j, num):
                        return False

                cells_in_col: list[int] = [r for r in range(9) if board[r][i] == 0 and num in candidates[r][i]]
                if len(cells_in_col) == 1:
                    r: int = cells_in_col[0]
                    board[r][i] = num
                    candidates[r][i] = set()
                    changed = True
                    if not eliminate(candidates, r, i, num):
                        return False

        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                for num in range(1, 10):
                    cells_in_box: list[tuple[int, int]] = [
                        (r, c) for r in range(box_row, box_row + 3)
                        for c in range(box_col, box_col + 3)
                        if board[r][c] == 0 and num in candidates[r][c]
                    ]
                    if len(cells_in_box) == 1:
                        r, c = cells_in_box[0]
                        board[r][c] = num
                        candidates[r][c] = set()
                        changed = True
                        if not eliminate(candidates, r, c, num):
                            return False
    return True


def eliminate(candidates: Candidates, row: int, col: int, num: int) -> bool:
    """Remove a placed digit from the candidates of all peers.

    Peers are cells sharing the same row, column, or 3x3 box as (row, col).
    If removing the last candidate from a peer that is not yet filled, the
    puzzle is unsolvable and False is returned.

    Args:
        candidates: Candidate grid (mutated in place).
        row:        Row index of the placed digit.
        col:        Column index of the placed digit.
        num:        The digit that was placed.

    Returns:
        True if all peers still have at least one candidate, False otherwise.
    """
    for j in range(9):
        if candidates[row][j] and num in candidates[row][j]:
            candidates[row][j].discard(num)
            if len(candidates[row][j]) == 0 and board_state(candidates, row, j) == 0:
                return False
    for i in range(9):
        if candidates[i][col] and num in candidates[i][col]:
            candidates[i][col].discard(num)
            if len(candidates[i][col]) == 0 and board_state(candidates, i, col) == 0:
                return False
    start_row: int = row - row % 3
    start_col: int = col - col % 3
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if candidates[r][c] and num in candidates[r][c]:
                candidates[r][c].discard(num)
                if len(candidates[r][c]) == 0 and board_state(candidates, r, c) == 0:
                    return False
    return True


def board_state(candidates: Candidates, row: int, col: int) -> int:
    """Return the only candidate for a cell, or 0 if still ambiguous.

    This is a helper used by eliminate() to detect cells that have been
    reduced to a single candidate but not yet committed to the board.

    Args:
        candidates: Candidate grid.
        row:        Row index.
        col:        Column index.

    Returns:
        The digit if exactly one candidate remains, otherwise 0.
    """
    if len(candidates[row][col]) == 1:
        return next(iter(candidates[row][col]))
    return 0


def is_solved(board: Board) -> bool:
    """Check whether every cell on the board has been filled.

    Args:
        board: Board to check.

    Returns:
        True if no zeros remain, False otherwise.
    """
    for row in board:
        if any(cell == 0 for cell in row):
            return False
    return True


def find_best_cell(candidates: Candidates) -> Optional[tuple[int, int]]:
    """Find the empty cell with the fewest remaining candidates (MRV heuristic).

    The cell with the smallest candidate set (> 1) is chosen for branching
    during backtracking, which minimises the branching factor.

    Args:
        candidates: Candidate grid.

    Returns:
        (row, col) of the best cell, or None if every cell is solved (all
        candidate sets are empty).
    """
    min_count: int = 10
    best: Optional[tuple[int, int]] = None
    for i in range(9):
        for j in range(9):
            count: int = len(candidates[i][j])
            if 1 < count < min_count:
                min_count = count
                best = (i, j)
    return best


def backtrack(board: Board, candidates: Candidates) -> Optional[Board]:
    """Depth-first search with MRV heuristic and constraint propagation.

    Picks the unfilled cell with the fewest candidates, tries each candidate
    in order, and recurses.  After placing a candidate, constraint propagation
    is re-run; if it leads to a contradiction the guess is undone and the next
    value is tried.

    Args:
        board:      Board being solved (mutated across recursive calls).
        candidates: Candidate grid (snapshotted/restored on backtrack).

    Returns:
        The solved board, or None if no assignment leads to a solution.
    """
    cell: Optional[tuple[int, int]] = find_best_cell(candidates)
    if not cell:
        return board if is_solved(board) else None

    row, col = cell
    saved_board: Board = [row[:] for row in board]

    for val in sorted(candidates[row][col]):
        saved_candidates: Candidates = [[set(c) for c in row] for row in candidates]

        board[row][col] = val
        candidates[row][col] = set()
        if eliminate(candidates, row, col, val) and constrain(board, candidates):
            if is_solved(board):
                return board
            result: Optional[Board] = backtrack(board, candidates)
            if result is not None:
                return result

        for i in range(9):
            for j in range(9):
                board[i][j] = saved_board[i][j]
        candidates = saved_candidates

    return None


def parse_board(puzzle_str: str) -> Board:
    """Parse a puzzle string into a 9x9 board.

    Accepts multiple formats:
      - Space-delimited rows (0 or . for blanks)
      - Grids with | / - border characters (stripped)
      - Single-line dot notation (81 consecutive characters)
      - Lines starting with # (ignored as comments)

    Args:
        puzzle_str: Raw puzzle text.

    Returns:
        A 9x9 list-of-lists of ints where 0 denotes an empty cell.
    """
    lines: list[str] = puzzle_str.splitlines()
    clean_lines: list[str] = []
    for line in lines:
        stripped: str = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        clean_lines.append(stripped)

    cleaned: str = ' '.join(clean_lines)
    cleaned = cleaned.replace('|', '').replace('-', '')

    chars: list[int] = []
    for char in cleaned:
        if char in '0.':
            chars.append(0)
        elif char.isdigit():
            chars.append(int(char))

    board: Board = []
    for i in range(9):
        row: list[int] = chars[i * 9:(i + 1) * 9]
        board.append(row)

    return board


def solve_from_str(puzzle_str: str) -> Optional[Board]:
    """Convenience wrapper: parse a puzzle string and solve it.

    Args:
        puzzle_str: Raw puzzle text in any supported format.

    Returns:
        Solved board or None.
    """
    board: Board = parse_board(puzzle_str)
    solution: Optional[Board] = solve_sudoku(board)
    return solution


def solve_from_file(filepath: str) -> None:
    """Read a puzzle file, solve it, and print the solution to stdout.

    Args:
        filepath: Path to a puzzle text file.
    """
    with open(filepath, 'r') as f:
        puzzle: str = f.read()

    solution: Optional[Board] = solve_from_str(puzzle)

    if solution is None:
        print("No solution exists")
        return

    for row in solution:
        print(' '.join(map(str, row)))


def main() -> None:
    """CLI entrypoint.

    Usage:
        python sudoku_vibes.py <puzzle_file>   solve a file
        python sudoku_vibes.py                  read puzzle from stdin
    """
    import sys
    if len(sys.argv) == 2:
        solve_from_file(sys.argv[1])
    elif len(sys.argv) == 1:
        puzzle: str = sys.stdin.read()
        solution: Optional[Board] = solve_from_str(puzzle)
        if solution is None:
            print("No solution exists")
            return
        for row in solution:
            print(' '.join(map(str, row)))
    else:
        print("Usage: sudoku_vibes.py [<puzzle_file>]")


if __name__ == '__main__':
    main()
