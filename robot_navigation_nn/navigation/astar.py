import heapq
import math

_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]

def astar(grid, start, goal):
    h, w = grid.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and
            0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if grid[start] or grid[goal]:
        return None
    frontier = [(0.0, start)]
    came_from = {start: None}
    cost = {start: 0.0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))
        for dr, dc, step_cost in _NEIGHBORS:
            nxt = (current[0] + dr, current[1] + dc)
            r, c = nxt
            if not (0 <= r < h and 0 <= c < w) or grid[r, c]:
                continue
            # Prevent diagonal corner cutting.
            if dr and dc and (grid[current[0] + dr, current[1]] or
                              grid[current[0], current[1] + dc]):
                continue
            new_cost = cost[current] + step_cost
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                heuristic = math.hypot(goal[0]-r, goal[1]-c)
                heapq.heappush(frontier, (new_cost + heuristic, nxt))
                came_from[nxt] = current
    return None
