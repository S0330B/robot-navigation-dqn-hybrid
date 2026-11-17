"""Headless DQN training with obstacle-aware (A*-distance) rewards and observations, guided exploration
and a density curriculum.

Examples:
  python -m ai.train --episodes 800 --environment static     # recommended first run
  python -m ai.train --episodes 1500 --environment both      # then add moving obstacles
Training resumes from checkpoints/dqn_checkpoint.pt when present.
"""
import argparse, math, random
import numpy as np
import torch
from config import Config
from environment.world import generate_complex_world, cell_center, make_observation, segment_collision, wrap_angle
from environment.dynamics import DynamicObstacles
from navigation.guidance import Guidance, guided_action
from ai.agent import DQNAgent
from ai.rewards import progress_reward

def run_training(episodes=800, environment='static', seed=7, max_steps=600, save_every=25,
                 epsilon_decay=None, guide_prob=None):
    cfg = Config(); rng = random.Random(seed); np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(device=device, epsilon_decay=epsilon_decay, hint_prob=guide_prob)
    turns = np.radians(np.asarray(cfg.action_turns_deg, dtype=np.float32))
    history = []

    for ep in range(1, episodes + 1):
        # Curriculum: lighter maps first, reaching the full density after 40% of the episodes.
        density = min(cfg.obstacle_density, 0.06 + 0.10 * min(1.0, ep / max(1, episodes * 0.4)))
        # In 'both' mode moving obstacles only appear after the first 30% of training.
        dynamic = environment == 'dynamic' or (environment == 'both' and ep > 0.3 * episodes and ep % 2 == 0)
        base, start, goal_cell, _ = generate_complex_world(cfg.map_size, rng)
        movers = DynamicObstacles(count=cfg.dynamic_count, seed=seed + ep); movers.reset(base, start, goal_cell)
        grid = movers.update(base, 0.0) if dynamic else base.copy()
        guide = Guidance(grid, goal_cell, cfg.guide_lookahead)
        pos = cell_center(start); heading = 0.0; goal = cell_center(goal_cell)
        obs = make_observation(grid, pos, heading, goal, cfg, guide)
        total = 0.0; success = False; collided = False

        for step in range(max_steps):
            hint = guided_action(obs, cfg.action_turns_deg)
            action = agent.act(obs, training=True, hint=hint)
            new_heading = wrap_angle(heading + float(turns[action]))
            candidate = pos + np.array([math.cos(new_heading), math.sin(new_heading)], dtype=np.float32) * cfg.move_step
            next_grid = movers.update(base, 0.12) if dynamic else base
            if dynamic: guide.update(next_grid)          # keep the path field consistent with the new grid
            collision = segment_collision(next_grid, pos, candidate, cfg.robot_radius)

            if collision:
                reward = cfg.reward_collision; next_pos = pos; done = True
            else:
                next_pos = candidate
                done = float(np.linalg.norm(goal - candidate)) <= cfg.goal_tolerance
                reward = progress_reward(guide.potential(pos), guide.potential(candidate), done, cfg)

            next_obs = make_observation(next_grid, next_pos, new_heading, goal, cfg, guide)
            agent.remember(obs, action, reward, next_obs, done)
            agent.learn(); total += reward
            pos = next_pos; heading = new_heading; grid = next_grid; obs = next_obs
            if done:
                collided = collision; success = not collision; break

        agent.episodes += 1; history.append((success, collided, total, step + 1))
        if ep % 10 == 0:
            recent = history[-50:]
            print(f"Episode {ep:4d}/{episodes} | density={density:.3f} | "
                  f"success(last {len(recent)})={sum(x[0] for x in recent)/len(recent):.0%} | "
                  f"collisions={sum(x[1] for x in recent)/len(recent):.0%} | eps={agent.epsilon:.3f} | "
                  f"replay={len(agent.memory)} | device={device}")
        if ep % save_every == 0: agent.save()

    agent.save()
    recent = history[-min(100, len(history)):]
    print(f"Training complete. Last {len(recent)} success rate (with exploration noise): "
          f"{sum(x[0] for x in recent)/len(recent):.1%}")
    print("For the real greedy score run:  python -m ai.evaluate --maps 100")
    print(f"Checkpoint + replay saved: {agent.checkpoint}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=800)
    parser.add_argument('--environment', choices=['static', 'dynamic', 'both'], default='static')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--epsilon-decay', type=float, default=None, help='per-learning-step decay (default from config)')
    parser.add_argument('--guide-prob', type=float, default=None, help='share of exploratory actions that follow A* waypoint')
    args = parser.parse_args()
    run_training(args.episodes, args.environment, args.seed, args.max_steps, args.save_every,
                 args.epsilon_decay, args.guide_prob)
