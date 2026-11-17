from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from config import Config
from ai.model import DQN
from ai.replay import ReplayBuffer

OBS_VERSION = 2

class DQNAgent:
    def __init__(self, device=None, checkpoint='checkpoints/dqn_checkpoint.pt',
                 lr=None, gamma=None, batch_size=None, capacity=100000,
                 epsilon_decay=None, epsilon_min=None, hint_prob=None):
        cfg = Config()
        lr = cfg.learning_rate if lr is None else lr
        self.gamma = cfg.gamma if gamma is None else gamma
        self.batch_size = cfg.batch_size if batch_size is None else batch_size
        self.epsilon_decay = cfg.epsilon_decay if epsilon_decay is None else epsilon_decay
        self.epsilon_min = cfg.epsilon_min if epsilon_min is None else epsilon_min
        self.hint_prob = cfg.guide_prob if hint_prob is None else hint_prob
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.online = DQN(hidden=cfg.hidden_sizes).to(self.device)
        self.target = DQN(hidden=cfg.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity)
        self.epsilon = 1.0
        self.learn_steps = 0; self.episodes = 0; self.checkpoint = Path(checkpoint)
        self.load()

    def act(self, state, training=True, hint=None):
        """Epsilon-greedy. While exploring, a share (`hint_prob`) of the random picks follow `hint`
        (the action that points at the A* waypoint) so replay contains many useful transitions."""
        if training and random.random() < self.epsilon:
            if hint is not None and random.random() < self.hint_prob:
                return int(hint)
            return random.randrange(5)
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(x).argmax(1).item())

    def remember(self, *transition): self.memory.add(*transition)

    def learn(self):
        if len(self.memory) < max(self.batch_size, 1000): return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        s = torch.as_tensor(states, device=self.device); a = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, device=self.device); ns = torch.as_tensor(next_states, device=self.device)
        d = torch.as_tensor(dones, device=self.device)
        q = self.online(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(ns).argmax(1, keepdim=True)
            next_q = self.target(ns).gather(1, next_actions).squeeze(1)
            target = r + self.gamma * (1 - d) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0); self.optimizer.step()
        self.learn_steps += 1; self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.learn_steps % 500 == 0: self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self):
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'obs_version': OBS_VERSION, 'online': self.online.state_dict(), 'target': self.target.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'epsilon': self.epsilon,
                    'learn_steps': self.learn_steps, 'episodes': self.episodes,
                    'replay': self.memory.state_dict()}, self.checkpoint)

    def load(self):
        if not self.checkpoint.exists(): return False
        try:
            data = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            if data.get('obs_version') != OBS_VERSION:
                print(f'Ignoring {self.checkpoint}: it was trained with an older observation/reward design. '
                      f'Starting a fresh DQN (delete the old file to silence this).')
                return False
            self.online.load_state_dict(data['online']); self.target.load_state_dict(data['target'])
            self.optimizer.load_state_dict(data['optimizer']); self.epsilon = float(data.get('epsilon', 1.0))
            self.learn_steps = int(data.get('learn_steps', 0)); self.episodes = int(data.get('episodes', 0))
            self.memory.load_state_dict(data.get('replay', [])); return True
        except Exception as exc:
            print(f'Checkpoint could not be restored ({exc}); starting a fresh DQN.')
            return False
