
def progress_reward(old_potential, new_potential, reached, cfg):
    """Potential-style progress along the obstacle-aware path, small step cost, goal bonus."""
    reward = cfg.reward_progress * (old_potential - new_potential) + cfg.reward_step
    if reached:
        reward += cfg.reward_goal
    return reward
