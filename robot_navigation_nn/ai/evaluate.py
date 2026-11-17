import argparse, math, random
import numpy as np
import torch
from config import Config
from environment.world import generate_world, cell_center, make_observation, segment_collision, wrap_angle
from navigation.controller import PathFollower
from navigation.guidance import Guidance, guided_action
from ai.agent import DQNAgent

def run_one(grid, start, goal_cell, mode, cfg, agent, max_steps=1200):
    pos = cell_center(start); goal = cell_center(goal_cell); heading = 0.0
    follower = PathFollower(grid, start, goal_cell, cfg); guide = Guidance(grid, goal_cell, cfg.guide_lookahead)
    turns = np.radians(np.asarray(cfg.action_turns_deg, dtype=np.float32)); length = 0.0
    for step in range(max_steps):
        old = pos.copy()
        if float(np.linalg.norm(goal - pos)) <= cfg.goal_tolerance: return True, False, length, step
        if mode == 'astar':
            pos, heading, reached, blocked = follower.update(pos, heading, 0.12)
            if blocked: follower.replan(pos)
            if reached: return True, False, length, step + 1
        else:
            obs = make_observation(grid, pos, heading, goal, cfg, guide)
            action = guided_action(obs, cfg.action_turns_deg) if mode == 'waypoint_ref' else agent.act(obs, training=False)
            nh = wrap_angle(heading + float(turns[action]))
            candidate = pos + np.array([math.cos(nh), math.sin(nh)], dtype=np.float32) * cfg.move_step
            collision = segment_collision(grid, pos, candidate, cfg.robot_radius)
            worse = guide.potential(candidate) > guide.potential(pos) + cfg.poor_action_margin
            if collision and mode in ('pure_dqn', 'waypoint_ref'):
                return False, True, length, step + 1        # only a real collision ends a pure run
            if mode == 'hybrid' and (collision or worse):
                pos, heading, reached, blocked = follower.update(pos, heading, 0.12)
                if blocked: follower.replan(pos)
                if reached: return True, False, length, step + 1
            else: pos = candidate; heading = nh
        length += float(np.linalg.norm(pos - old))
    return False, False, length, max_steps

def main():
    p = argparse.ArgumentParser(); p.add_argument('--maps', type=int, default=100); p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args(); cfg = Config(); rng = random.Random(args.seed)
    agent = DQNAgent(device='cuda' if torch.cuda.is_available() else 'cpu')
    modes = ['pure_dqn', 'hybrid', 'astar', 'waypoint_ref']; results = {m: [] for m in modes}
    for i in range(args.maps):
        grid, start, goal, path = generate_world(cfg.map_size, cfg.obstacle_density, rng)
        for mode in modes: results[mode].append(run_one(grid, start, goal, mode, cfg, agent))
    print('Mode          Success%  Collision%  Mean path length  Mean steps')
    for mode, rows in results.items():
        print(f"{mode:13s} {100*np.mean([x[0] for x in rows]):7.1f} {100*np.mean([x[1] for x in rows]):10.1f} {np.mean([x[2] for x in rows]):17.2f} {np.mean([x[3] for x in rows]):11.1f}")
    print('All methods used the same generated maps. pure_dqn has no A* intervention (only a collision ends it).')
    print('waypoint_ref is a blind steering rule with no network - a sanity reference, not a trained policy.')
if __name__ == '__main__': main()
