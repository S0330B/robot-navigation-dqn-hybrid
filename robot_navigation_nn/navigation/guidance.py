import heapq
import math
import numpy as np

_SQ2 = math.sqrt(2.0)
_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, _SQ2), (-1, 1, _SQ2), (1, -1, _SQ2), (1, 1, _SQ2)]


def _neighbors(grid, r, c):
    h, w = grid.shape
    for dr, dc, cost in _MOVES:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc]:
            continue
        if dr and dc and (grid[r + dr, c] or grid[r, c + dc]):
            continue
        yield nr, nc, cost


def distance_field(grid, goal_cell):
    h, w = grid.shape
    field = np.full((h, w), np.inf, dtype=np.float64)
    if grid[goal_cell]:
        return field
    field[goal_cell] = 0.0
    heap = [(0.0, goal_cell)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if d > field[r, c]:
            continue
        for nr, nc, cost in _neighbors(grid, r, c):
            nd = d + cost
            if nd < field[nr, nc]:
                field[nr, nc] = nd
                heapq.heappush(heap, (nd, (nr, nc)))
    return field


class Guidance:
    def __init__(self, grid, goal_cell, lookahead=1):
        self.goal_cell = tuple(goal_cell)
        self.lookahead = int(lookahead)
        self.goal = np.array([goal_cell[1] + 0.5, goal_cell[0] + 0.5], dtype=np.float32)
        self.grid = None
        self.field = None
        self.update(grid)

    def update(self, grid):
        """Recompute the field only if the grid changed (moving obstacles). Returns True if recomputed."""
        if self.grid is not None and np.array_equal(grid, self.grid):
            return False
        self.grid = np.array(grid, copy=True)
        self.field = distance_field(self.grid, self.goal_cell)
        return True

    def _cell(self, pos):
        h, w = self.grid.shape
        return (min(max(int(pos[1]), 0), h - 1), min(max(int(pos[0]), 0), w - 1))

    def distance(self, pos):
        """Smooth path distance from a continuous position to the goal (inf if unreachable)."""
        r, c = self._cell(pos)
        best = math.inf
        options = [(r, c)] + [(nr, nc) for nr, nc, _ in _neighbors(self.grid, r, c)]
        for nr, nc in options:
            f = self.field[nr, nc]
            if math.isfinite(f):
                best = min(best, f + math.hypot(pos[0] - (nc + 0.5), pos[1] - (nr + 0.5)))
        return best

    def potential(self, pos):
        """Path distance, falling back to straight-line distance if the goal is currently unreachable."""
        d = self.distance(pos)
        if math.isfinite(d):
            return d
        return float(np.linalg.norm(self.goal - np.asarray(pos, dtype=np.float32)))

    def waypoint(self, pos):
        """A point `lookahead` cells ahead along the shortest path, or the goal if unreachable/close."""
        r, c = self._cell(pos)
        if not math.isfinite(self.field[r, c]):
            return self.goal
        for _ in range(self.lookahead):
            if (r, c) == self.goal_cell:
                return self.goal
            best = (self.field[r, c], r, c)
            for nr, nc, _ in _neighbors(self.grid, r, c):
                if self.field[nr, nc] < best[0]:
                    best = (self.field[nr, nc], nr, nc)
            r, c = best[1], best[2]
        if (r, c) == self.goal_cell:
            return self.goal
        return np.array([c + 0.5, r + 0.5], dtype=np.float32)


def guided_action(obs, action_turns_deg):
    """Discrete steering action that best points at the waypoint (obs[-2:] = sin, cos of bearing)."""
    bearing = math.atan2(float(obs[-2]), float(obs[-1]))
    turns = np.radians(np.asarray(action_turns_deg, dtype=np.float32))
    err = np.abs((bearing - turns + np.pi) % (2 * np.pi) - np.pi)
    return int(np.argmin(err))
