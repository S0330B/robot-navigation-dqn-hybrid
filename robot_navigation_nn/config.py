from dataclasses import dataclass

@dataclass
class Config:
    map_size: int = 24
    obstacle_density: float = 0.16
    robot_radius: float = 0.18
    move_step: float = 0.28
    action_turns_deg: tuple = (-45.0, -22.5, 0.0, 22.5, 45.0)
    goal_tolerance: float = 0.48
    max_episode_steps: int = 1200
    lidar_rays: int = 24
    lidar_max_range: float = 7.0
    cell_pixels: int = 27
    fps: int = 60
    ai_decision_interval: float = 0.12
    manual_speed: float = 2.6
    manual_turn_speed: float = 2.8
    seed: int = 7
    # --- DQN ---
    hidden_sizes: tuple = (256, 128)
    learning_rate: float = 0.0003
    batch_size: int = 64
    gamma: float = 0.99
    epsilon_decay: float = 0.99995      # per learning step: ~60k steps from 1.0 to 0.05
    epsilon_min: float = 0.05
    guide_prob: float = 0.6             # share of exploratory actions that follow the A* waypoint
    # --- guidance / rewards (progress is measured along the obstacle-aware path) ---
    guide_lookahead: int = 1
    reward_progress: float = 1.0
    reward_step: float = -0.02
    reward_goal: float = 20.0
    reward_collision: float = -20.0
    poor_action_margin: float = 0.1     # hybrid: path distance may grow this much before A* takes over
    dynamic_count: int = 5
