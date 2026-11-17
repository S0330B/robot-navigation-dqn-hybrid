import math
import random
from pathlib import Path
import pygame
import numpy as np

from config import Config
from environment.world import generate_world, cell_center, point_collision, segment_collision, lidar_readings, make_observation, wrap_angle
from environment.dynamics import DynamicObstacles
from ai.agent import DQNAgent
from navigation.controller import PathFollower
from navigation.astar import astar
from navigation.guidance import Guidance, guided_action
from ai.rewards import progress_reward

try:
    import torch
except Exception:
    torch = None

WIDTH, HEIGHT = 1120, 780
PANEL_W = 360
BG = (17, 22, 31)
PANEL = (27, 34, 46)
GRID = (47, 57, 71)
WALL = (94, 111, 132)
ROBOT = (63, 190, 255)
GOAL = (70, 220, 130)
PATH = (245, 190, 70)
WHITE = (235, 240, 247)
MUTED = (155, 169, 188)
RED = (245, 90, 90)

def draw_text(surface, font, text, pos, color=WHITE):
    surface.blit(font.render(str(text), True, color), pos)

class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Robot Navigation Lab | AI • Manual • Live Simulator")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 27)
        self.small = pygame.font.Font(None, 21)
        self.big = pygame.font.Font(None, 48)
        self.cfg = Config()
        self.rng = random.Random(self.cfg.seed)
        self.mode = "menu"
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.agent = DQNAgent(device=self.device)
        self.dynamic_enabled = False
        self.dynamic_obstacles = DynamicObstacles(count=5, seed=self.cfg.seed)
        self.episode_reward = 0.0
        self.last_loss = None
        self.new_world()
        self.running = True
        self.ai_timer = 0.0
        self.ai_steps = 0
        self.status = "READY"
        self.message = "Choose a mode from the main menu."
        self.live_success = 0
        self.live_total = 0
        self.live_collisions = 0
        self.show_lidar = True
        self.paused = False
        self.last_action = "—"

    def new_world(self):
        self.static_grid, self.start_cell, self.goal_cell, self.path = generate_world(
            self.cfg.map_size, self.cfg.obstacle_density, self.rng)
        self.dynamic_obstacles.reset(self.static_grid, self.start_cell, self.goal_cell)
        self.grid = self.dynamic_obstacles.update(self.static_grid, 0.0) if self.dynamic_enabled else self.static_grid.copy()
        self.position = cell_center(self.start_cell)
        self.heading = 0.0
        self.goal = cell_center(self.goal_cell)
        self.follower = PathFollower(self.grid, self.start_cell, self.goal_cell, self.cfg)
        self.guide = Guidance(self.grid, self.goal_cell, self.cfg.guide_lookahead)
        self.trajectory = [self.position.copy()]
        self.ai_steps = 0
        self.ai_timer = 0.0
        self.episode_reward = 0.0
        self.status = "RUNNING"
        self.paused = False
        self.last_action = "—"

    def set_mode(self, mode):
        self.mode = mode
        self.new_world()
        self.message = {
            "ai": "Autonomous AI navigation. A* safety recovery is enabled.",
            "manual": "Use arrows or WASD to drive. Reach the green goal.",
            "live": "Continuous live simulation: autonomous runs on fresh maps."
        }[mode]

    def finish_run(self, success, collision=False):
        # Every terminal episode updates the persistent DQN statistics.
        if self.mode in ("ai", "live"):
            self.agent.episodes += 1
        if self.mode == "live":
            self.live_total += 1
            if success: self.live_success += 1
            if collision: self.live_collisions += 1
            self.agent.save()
            self.new_world()
        else:
            self.agent.save()
            self.status = "GOAL REACHED" if success else ("COLLISION" if collision else "STOPPED")

    def ai_update(self, dt, use_model=True):
        if self.status != "RUNNING" or self.paused: return
        # Moving obstacles update in the dynamic environment. Static maps remain unchanged.
        if self.dynamic_enabled:
            old_grid = self.grid
            self.grid = self.dynamic_obstacles.update(self.static_grid, dt)
            self.follower.grid = self.grid
            self.guide.update(self.grid)
            # Replan if a moving obstacle has appeared on the current route.
            if np.any((self.grid != 0) & (old_grid == 0)):
                self.follower.replan(self.position)
        self.ai_timer += dt
        if self.ai_timer < self.cfg.ai_decision_interval: return
        step_dt = self.ai_timer; self.ai_timer = 0.0; self.ai_steps += 1
        obs = make_observation(self.grid, self.position, self.heading, self.goal, self.cfg, self.guide)
        old_position = self.position.copy()
        old_potential = self.guide.potential(self.position)
        hint = guided_action(obs, self.cfg.action_turns_deg)
        action = self.agent.act(obs, training=(self.mode in ("ai", "live")), hint=hint) if use_model else None
        proposed_ok = False
        if action is not None:
            turn = math.radians(self.cfg.action_turns_deg[action])
            candidate_heading = wrap_angle(self.heading + turn)
            candidate = self.position + np.array([math.cos(candidate_heading), math.sin(candidate_heading)], dtype=np.float32) * self.cfg.move_step
            collision = segment_collision(self.grid, self.position, candidate, self.cfg.robot_radius)
            new_potential = self.guide.potential(candidate)
            # Penalize unsafe actions even though the hybrid safety layer prevents a crash.
            if collision:
                self.agent.remember(obs, action, self.cfg.reward_collision, obs, False)
                self.last_action = f"Unsafe DQN action {action} (penalized)"
                self.last_loss = self.agent.learn()
            elif new_potential > old_potential + self.cfg.poor_action_margin:
                reward = progress_reward(old_potential, new_potential, False, self.cfg)
                self.agent.remember(obs, action, reward, obs, False)
                self.last_action = f"Poor DQN action {action} (A* recovery)"
                self.last_loss = self.agent.learn()
            else:
                self.position = candidate; self.heading = candidate_heading
                done = float(np.linalg.norm(self.goal - candidate)) <= self.cfg.goal_tolerance
                reward = progress_reward(old_potential, new_potential, done, self.cfg)
                proposed_ok = True
                self.last_action = f"DQN action {action}"
                next_obs = make_observation(self.grid, self.position, self.heading, self.goal, self.cfg, self.guide)
                self.agent.remember(obs, action, reward, next_obs, done)
                self.last_loss = self.agent.learn()
                self.episode_reward += reward
                if done:
                    self.finish_run(True); return
        if not proposed_ok:
            # A* is a safety/recovery controller, not credited as a DQN action.
            self.follower.grid = self.grid
            self.position, self.heading, reached, blocked = self.follower.update(self.position, self.heading, step_dt)
            if blocked: self.follower.replan(self.position)
            self.last_action += " | A* safety" if action is not None else "A* fallback"
            if reached or np.linalg.norm(self.goal-self.position) <= self.cfg.goal_tolerance:
                self.finish_run(True); return
        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > 3000: self.trajectory = self.trajectory[-2500:]
        if self.ai_steps >= self.cfg.max_episode_steps:
            self.agent.remember(obs, action if action is not None else 2, self.cfg.reward_collision / 2, obs, True)
            self.last_loss = self.agent.learn()
            self.finish_run(False)
            if self.mode == "ai":
                self.message = "Episode limit reached; restarting with learned experience retained."
                self.new_world()

    def manual_update(self, dt, keys):
        if self.status != "RUNNING" or self.paused:
            return
        turn = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            turn -= self.cfg.manual_turn_speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            turn += self.cfg.manual_turn_speed * dt
        self.heading = wrap_angle(self.heading + turn)
        direction = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            direction += 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            direction -= 1.0
        if direction:
            distance = direction * self.cfg.manual_speed * dt
            candidate = self.position + np.array([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32) * distance
            if not segment_collision(self.grid, self.position, candidate, self.cfg.robot_radius):
                self.position = candidate
                self.trajectory.append(self.position.copy())
                self.last_action = "Forward" if direction > 0 else "Reverse"
            else:
                self.last_action = "Blocked by obstacle"
        if np.linalg.norm(self.goal-self.position) <= self.cfg.goal_tolerance:
            self.finish_run(True)

    def draw(self):
        self.screen.fill(BG)
        cell = self.cfg.cell_pixels
        ox, oy = 26, 54
        board = self.cfg.map_size * cell
        # Map background and grid.
        pygame.draw.rect(self.screen, (21, 28, 39), (ox, oy, board, board), border_radius=5)
        for r in range(self.cfg.map_size):
            for c in range(self.cfg.map_size):
                rect = pygame.Rect(ox+c*cell, oy+r*cell, cell, cell)
                if self.grid[r,c]:
                    pygame.draw.rect(self.screen, WALL, rect)
                pygame.draw.rect(self.screen, GRID, rect, 1)
        # A* route.
        if self.path:
            pts = [(ox+int((c+0.5)*cell), oy+int((r+0.5)*cell)) for r,c in self.path]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (112, 100, 73), False, pts, 3)
        # Robot trajectory.
        if len(self.trajectory) > 1:
            pts = [(ox+int(p[0]*cell), oy+int(p[1]*cell)) for p in self.trajectory]
            pygame.draw.lines(self.screen, PATH, False, pts, 2)
        # Goal.
        gx, gy = ox+int(self.goal[0]*cell), oy+int(self.goal[1]*cell)
        pygame.draw.circle(self.screen, GOAL, (gx,gy), int(cell*0.34))
        pygame.draw.circle(self.screen, (12,40,27), (gx,gy), int(cell*0.34), 2)
        # LiDAR rays.
        if self.show_lidar and self.mode in ("ai","live"):
            readings = lidar_readings(self.grid, self.position, self.heading,
                                      self.cfg.lidar_rays, self.cfg.lidar_max_range)
            cx, cy = ox+int(self.position[0]*cell), oy+int(self.position[1]*cell)
            for i, value in enumerate(readings):
                angle = self.heading + 2*math.pi*i/self.cfg.lidar_rays
                length = float(value)*self.cfg.lidar_max_range*cell
                end = (cx+int(math.cos(angle)*length), cy+int(math.sin(angle)*length))
                pygame.draw.line(self.screen, (80,125,155), (cx,cy), end, 1)
        # Robot body and heading.
        rx, ry = ox+int(self.position[0]*cell), oy+int(self.position[1]*cell)
        radius = max(6, int(self.cfg.robot_radius*cell))
        pygame.draw.circle(self.screen, ROBOT, (rx,ry), radius)
        nose = (rx+int(math.cos(self.heading)*radius*1.8), ry+int(math.sin(self.heading)*radius*1.8))
        pygame.draw.line(self.screen, WHITE, (rx,ry), nose, 3)

        # Side panel.
        px = ox + board + 24
        pygame.draw.rect(self.screen, PANEL, (px, 20, WIDTH-px-20, HEIGHT-40), border_radius=12)
        title = {"menu":"Robot Navigation Lab", "ai":"AI MODE", "manual":"MANUAL MODE", "live":"LIVE SIMULATOR"}[self.mode]
        draw_text(self.screen, self.big if self.mode=="menu" else self.font, title, (px+20, 42))
        draw_text(self.screen, self.small, self.message, (px+20, 100), MUTED)
        draw_text(self.screen, self.font, f"Status: {self.status}", (px+20, 150),
                  GOAL if self.status=="GOAL REACHED" else WHITE)
        draw_text(self.screen, self.small, f"Steps: {self.ai_steps}", (px+20, 190), MUTED)
        draw_text(self.screen, self.small, f"Distance to goal: {np.linalg.norm(self.goal-self.position):.2f}", (px+20, 220), MUTED)
        draw_text(self.screen, self.small, f"Last action: {self.last_action}", (px+20, 250), MUTED)
        if self.mode == "ai":
            model_text = f"DQN {self.device.upper()} | eps {self.agent.epsilon:.2f} | replay {len(self.agent.memory)}"
            draw_text(self.screen, self.small, model_text, (px+20, 285), GOAL)
            draw_text(self.screen, self.small, f"Episodes {self.agent.episodes} | updates {self.agent.learn_steps}", (px+20, 315), MUTED)
            draw_text(self.screen, self.small, f"Environment: {'DYNAMIC' if self.dynamic_enabled else 'STATIC'} (E to switch)", (px+20, 345), PATH)
        if self.mode == "live":
            rate = self.live_success / max(1,self.live_total)
            draw_text(self.screen, self.small, f"Completed runs: {self.live_total}", (px+20, 285), MUTED)
            draw_text(self.screen, self.small, f"Successes: {self.live_success} | Success rate: {rate:.1%}", (px+20, 315), GOAL)
            draw_text(self.screen, self.small, f"Collisions: {self.live_collisions}", (px+20, 345), MUTED)
        if self.mode == "menu":
            self.draw_menu(px)
        else:
            y = 390 if self.mode == "live" else 335
            controls = [
                "Controls",
                "SPACE  Pause / resume",
                "N      Generate a new map",
                "L      Toggle LiDAR display",
                "E      Toggle static / dynamic map",
                "S      Save DQN learning now",
                "M      Return to mode menu",
                "ESC    Quit",
            ]
            if self.mode == "manual":
                controls[1] = "ARROWS / WASD  Drive and turn"
                controls.insert(2, "SPACE          Pause / resume")
            for line in controls:
                draw_text(self.screen, self.small, line, (px+20,y), WHITE if line=="Controls" else MUTED)
                y += 29
        pygame.display.flip()

    def draw_menu(self, px):
        y = 190
        options = [
            ("1  AI MODE", "Autonomous learned policy with A* safety recovery"),
            ("2  MANUAL MODE", "Drive the robot with keyboard controls"),
            ("3  LIVE SIMULATOR", "Watch repeated autonomous runs and live statistics"),
        ]
        for title, desc in options:
            pygame.draw.rect(self.screen, (42,54,71), (px+18,y,WIDTH-px-56,82), border_radius=9)
            draw_text(self.screen, self.font, title, (px+34,y+12))
            draw_text(self.screen, self.small, desc, (px+34,y+47), MUTED)
            y += 100
        draw_text(self.screen, self.small, "DQN learns online; checkpoints persist automatically.", (px+20,y+8), PATH)
        draw_text(self.screen, self.small, "E toggles static / moving obstacles in AI and Live.", (px+20,y+38), MUTED)

    def run(self):
        while self.running:
            dt = min(self.clock.tick(self.cfg.fps)/1000.0, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.mode in ("ai", "live"): self.agent.save()
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.mode in ("ai", "live"): self.agent.save()
                        self.running = False
                    elif self.mode == "menu":
                        if event.key == pygame.K_1: self.set_mode("ai")
                        elif event.key == pygame.K_2: self.set_mode("manual")
                        elif event.key == pygame.K_3: self.set_mode("live")
                    else:
                        if event.key == pygame.K_SPACE: self.paused = not self.paused
                        elif event.key == pygame.K_n: self.new_world()
                        elif event.key == pygame.K_l: self.show_lidar = not self.show_lidar
                        elif event.key == pygame.K_e:
                            self.dynamic_enabled = not self.dynamic_enabled
                            self.new_world()
                            self.message = "Dynamic obstacles enabled." if self.dynamic_enabled else "Static environment enabled."
                        elif event.key == pygame.K_s: self.agent.save(); self.message = "DQN checkpoint and replay memory saved."
                        elif event.key == pygame.K_m: self.mode = "menu"; self.message = "Choose a mode from the main menu."
            keys = pygame.key.get_pressed()
            if self.mode == "ai":
                self.ai_update(dt, use_model=True)
            elif self.mode == "live":
                self.ai_update(dt, use_model=True)
            elif self.mode == "manual":
                self.manual_update(dt, keys)
            self.draw()
        pygame.quit()

if __name__ == "__main__":
    App().run()
