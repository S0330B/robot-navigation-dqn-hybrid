import math
import numpy as np
from environment.world import cell_center, segment_collision, wrap_angle
from navigation.astar import astar

class PathFollower:
    def __init__(self, grid, start_cell, goal_cell, config):
        self.grid = grid
        self.goal_cell = goal_cell
        self.goal = cell_center(goal_cell)
        self.config = config
        self.path = astar(grid, start_cell, goal_cell) or []
        self.index = 1
        self.steps = 0
        self.stuck_ticks = 0
        self.position = cell_center(start_cell)

    def replan(self, position):
        cell = (int(position[1]), int(position[0]))
        cell = (max(0, min(self.grid.shape[0]-1, cell[0])),
                max(0, min(self.grid.shape[1]-1, cell[1])))
        if self.grid[cell]:
            candidates = []
            for r in range(max(1, cell[0]-2), min(self.grid.shape[0]-1, cell[0]+3)):
                for c in range(max(1, cell[1]-2), min(self.grid.shape[1]-1, cell[1]+3)):
                    if not self.grid[r, c]:
                        candidates.append(((r,c), (r+0.5-position[1])**2 + (c+0.5-position[0])**2))
            if not candidates:
                return False
            cell = min(candidates, key=lambda x:x[1])[0]
        self.path = astar(self.grid, cell, self.goal_cell) or []
        self.index = 1
        return bool(self.path)

    def target(self):
        if not self.path:
            return self.goal
        while self.index < len(self.path)-1:
            p = cell_center(self.path[self.index])
            if np.linalg.norm(p - self.position) > 0.35:
                break
            self.index += 1
        return cell_center(self.path[min(self.index, len(self.path)-1)])

    def update(self, position, heading, dt):
        self.position = np.asarray(position, dtype=np.float32)
        self.steps += 1
        if self.steps % 30 == 0:
            self.replan(self.position)
        if np.linalg.norm(self.goal - self.position) <= self.config.goal_tolerance:
            return self.position.copy(), heading, True, False
        if not self.path:
            self.replan(self.position)
        target = self.target()
        delta = target - self.position
        desired = math.atan2(float(delta[1]), float(delta[0]))
        error = wrap_angle(desired - heading)
        
        max_turn = 4.0 * dt
        new_heading = heading + max(-max_turn, min(max_turn, error))
        speed = 2.3
        distance = min(speed * dt, float(np.linalg.norm(delta)))
        candidate = self.position + np.array([math.cos(new_heading), math.sin(new_heading)], dtype=np.float32) * distance
        
        if segment_collision(self.grid, self.position, candidate, self.config.robot_radius):
            self.replan(self.position)
            new_heading = heading + max(-max_turn, min(max_turn, error))
            candidate = self.position + np.array([math.cos(new_heading), math.sin(new_heading)], dtype=np.float32) * (distance * 0.4)
            if segment_collision(self.grid, self.position, candidate, self.config.robot_radius):
                return self.position.copy(), new_heading, False, True
                
        moved = float(np.linalg.norm(candidate - self.position))
        if moved < 0.002:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        if self.stuck_ticks > 25:
            self.replan(self.position)
            self.stuck_ticks = 0
            
        return candidate, new_heading, False, False
