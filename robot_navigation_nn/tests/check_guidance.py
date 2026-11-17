
import argparse, math, random
import numpy as np
from config import Config
from environment.world import generate_world, cell_center, make_observation, segment_collision, wrap_angle
from environment.dynamics import DynamicObstacles
from navigation.guidance import Guidance, guided_action

def run(cfg, grid_base, start, goal_cell, dynamic, seed, use_guide=True, max_steps=900):
    movers = DynamicObstacles(count=cfg.dynamic_count, seed=seed); movers.reset(grid_base, start, goal_cell)
    grid = movers.update(grid_base, 0.0) if dynamic else grid_base.copy()
    guide = Guidance(grid, goal_cell, cfg.guide_lookahead) if use_guide else None
    pos = cell_center(start); goal = cell_center(goal_cell); heading = 0.0
    turns = np.radians(np.asarray(cfg.action_turns_deg, dtype=np.float32))
    for step in range(max_steps):
        obs = make_observation(grid, pos, heading, goal, cfg, guide)
        a = guided_action(obs, cfg.action_turns_deg)
        nh = wrap_angle(heading + float(turns[a]))
        cand = pos + np.array([math.cos(nh), math.sin(nh)], dtype=np.float32) * cfg.move_step
        if dynamic:
            grid = movers.update(grid_base, 0.12)
            if guide: guide.update(grid)
        if segment_collision(grid, pos, cand, cfg.robot_radius): return 'collision', step
        pos, heading = cand, nh
        if np.linalg.norm(goal - pos) <= cfg.goal_tolerance: return 'success', step
    return 'timeout', max_steps

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--maps', type=int, default=100); ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--lookahead', type=int, default=None)
    args = ap.parse_args(); cfg = Config()
    if args.lookahead is not None: cfg.guide_lookahead = args.lookahead
    for dynamic in (False, True):
        for use_guide in (False, True):
            rng = random.Random(args.seed); tally = {'success': 0, 'collision': 0, 'timeout': 0}; steps = []
            for i in range(args.maps):
                grid, start, goal, _ = generate_world(cfg.map_size, cfg.obstacle_density, rng)
                outcome, n = run(cfg, grid, start, goal, dynamic, args.seed + i, use_guide)
                tally[outcome] += 1
                if outcome == 'success': steps.append(n)
            name = ('dynamic' if dynamic else 'static') + (' | path-waypoint obs' if use_guide else ' | straight-line obs')
            print(f"{name:42s} success {tally['success']:3d}%  collision {tally['collision']:3d}%  timeout {tally['timeout']:3d}%  mean steps {np.mean(steps) if steps else float('nan'):.0f}"
                  if args.maps == 100 else f"{name:42s} {tally}")
main()
