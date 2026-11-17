import math
import random
import numpy as np
from navigation.astar import astar

def cell_center(cell):
    row, col = cell
    return np.array([col + 0.5, row + 0.5], dtype=np.float32)

def point_collision(grid, x, y, radius=0.18):
    h, w = grid.shape
    if x < radius or y < radius or x >= w - radius or y >= h - radius:
        return True
    r0 = max(0, int(math.floor(y - radius)))
    r1 = min(h - 1, int(math.floor(y + radius)))
    c0 = max(0, int(math.floor(x - radius)))
    c1 = min(w - 1, int(math.floor(x + radius)))
    rr = radius * radius
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            if grid[row, col]:
                nearest_x = min(max(x, col), col + 1.0)
                nearest_y = min(max(y, row), row + 1.0)
                if (x-nearest_x)**2 + (y-nearest_y)**2 <= rr:
                    return True
    return False

def segment_collision(grid, start, end, radius=0.18, resolution=0.04):
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    distance = float(np.linalg.norm(b-a))
    n = max(1, int(math.ceil(distance / resolution)))
    for i in range(1, n+1):
        p = a + (b-a) * (i/n)
        if point_collision(grid, float(p[0]), float(p[1]), radius):
            return True
    return False

def generate_world(size=24, density=0.16, rng=None):
    rng = rng or random.Random()
    for _ in range(300):
        grid = np.zeros((size, size), dtype=np.uint8)
        grid[0, :] = grid[-1, :] = 1
        grid[:, 0] = grid[:, -1] = 1
        # Mix small blocks and individual obstacles, leaving start/goal areas clear.
        for row in range(1, size-1):
            for col in range(1, size-1):
                if (row, col) in ((1, 1), (1, 2), (2, 1),
                                  (size-2, size-2), (size-3, size-2), (size-2, size-3)):
                    continue
                if rng.random() < density:
                    grid[row, col] = 1
        start = (1, 1)
        goal = (size-2, size-2)
        path = astar(grid, start, goal)
        if path and len(path) >= 5:
            return grid, start, goal, path
    # Guaranteed fallback: clear a simple diagonal corridor.
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[0, :] = grid[-1, :] = 1
    grid[:, 0] = grid[:, -1] = 1
    for i in range(1, size-1):
        if i not in (1, size-2):
            grid[i, i] = 1
    path = astar(grid, (1, 1), (size-2, size-2))
    return grid, (1, 1), (size-2, size-2), path

def lidar_readings(grid, position, heading, rays=24, max_range=7.0):
    values = []
    x, y = float(position[0]), float(position[1])
    h, w = grid.shape
    for i in range(rays):
        angle = heading + 2.0 * math.pi * i / rays
        distance = 0.0
        while distance < max_range:
            distance += 0.06
            px = x + math.cos(angle) * distance
            py = y + math.sin(angle) * distance
            if px < 0 or py < 0 or px >= w or py >= h:
                break
            if grid[int(py), int(px)]:
                break
        values.append(min(distance, max_range) / max_range)
    return np.asarray(values, dtype=np.float32)

def make_observation(grid, position, heading, goal, config, guide=None):
    """27 features: 24 LiDAR rays + [path distance, sin(bearing), cos(bearing)].

    With a `guide` (navigation.guidance.Guidance) the distance/bearing refer to the next
    waypoint on the obstacle-aware shortest path, so the policy knows which way to detour.
    Without it they fall back to the straight line to the goal.
    """
    lidar = lidar_readings(grid, position, heading,
                           config.lidar_rays, config.lidar_max_range)
    position = np.asarray(position, dtype=np.float32)
    target = goal
    distance = float(np.linalg.norm(goal - position))
    if guide is not None:
        path_distance = guide.distance(position)
        if math.isfinite(path_distance):
            target = guide.waypoint(position)
            distance = path_distance
    delta = target - position
    bearing = wrap_angle(math.atan2(float(delta[1]), float(delta[0])) - heading)
    scale = config.map_size * (2.0 if guide is not None else 1.414)
    goal_features = np.asarray([
        min(distance / scale, 1.0),
        math.sin(bearing), math.cos(bearing)
    ], dtype=np.float32)
    return np.concatenate([lidar, goal_features]).astype(np.float32)

def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def generate_complex_world(size=30, rng=None):
    rng = rng or random.Random()
    for _ in range(500):
        grid = np.zeros((size, size), dtype=np.uint8)

        # border walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        # add maze-like vertical walls
        for c in range(2, size - 2, 3):
            for r in range(1, size - 1):
                if rng.random() < 0.7:
                    grid[r, c] = 1

        # add maze-like horizontal walls
        for r in range(2, size - 2, 3):
            for c in range(1, size - 1):
                if rng.random() < 0.7:
                    grid[r, c] = 1

        # add cluster obstacles
        for _ in range(20):
            cx = rng.randint(4, size - 5)
            cy = rng.randint(4, size - 5)
            radius = rng.randint(2, 5)
            for y in range(max(1, cy-radius), min(size-1, cy+radius+1)):
                for x in range(max(1, cx-radius), min(size-1, cx+radius+1)):
                    if (x-cx)**2 + (y-cy)**2 <= radius**2:
                        grid[y, x] = 1

        # clear start and goal area
        start = (1, 1)
        goal = (size - 2, size - 2)
        for y in range(0, 4):
            for x in range(0, 4):
                grid[y, x] = 0
        for y in range(size - 4, size):
            for x in range(size - 4, size):
                grid[y, x] = 0

        grid[start] = 0
        grid[goal] = 0

        # ensure the robot has a route
        path = astar(grid, start, goal)
        if path and len(path) >= 10:
            return grid, start, goal, path

    # fallback simple map
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    start = (1, 1)
    goal = (size - 2, size - 2)
    return grid, start, goal, [(1,1), (1,2), (2,2), (3,3)]
