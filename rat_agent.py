import copy
import numpy as np
import random


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def angdiff(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


class RatAgent:
    def __init__(self, agent_id, start_pos, config, dna=None, rng_seed=None):
        # Lazy import to avoid circular dependency when app.py imports TournamentManager
        from app import DendriticCluster

        self.id = agent_id
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.heading = 0.0
        self.score = 0
        self.deaths = 0
        self.frames_alive = 0
        self.frustration = 0.0
        self.dopamine_tonic = float(self.config.get("dopamine_tonic", 0.2))
        self.dopamine_phasic = 0.0
        self.serotonin = 0.5
        self.panic = 0.0
        self.cortisol = 0.0
        self.brain_data = {}

        self.dna = dna if dna else {}
        self.config = copy.deepcopy(config)
        self.config.update(self.dna)

        self.rng = np.random.default_rng(rng_seed)
        seed_int = int(self.rng.integers(0, 100000))
        self.py_rng = random.Random(seed_int)

        self.brain = DendriticCluster(
            config=self.config,
            rng=self.rng,
            py_rng=self.py_rng,
            mt_rng=self.rng,
            load_genome=False
        )

        self.whisk_angle = 0.0
        self.whisk_phase = 0.0
        self.whisker_hits = [False, False]
        self.vision_buffer = []
        self.last_collision = False
        self.last_touch = 0.0

    def check_collision(self, pos, grid):
        """Return True if position (with small radius) hits a wall or is out of bounds."""
        map_h, map_w = grid.shape

        def is_wall(x, y):
            if x < 0 or x >= map_w or y < 0 or y >= map_h:
                return True
            return grid[int(y), int(x)] == 1

        if is_wall(pos[0], pos[1]):
            return True

        radius = 0.3
        if is_wall(pos[0] + radius, pos[1]):
            return True
        if is_wall(pos[0] - radius, pos[1]):
            return True
        if is_wall(pos[0], pos[1] + radius):
            return True
        if is_wall(pos[0], pos[1] - radius):
            return True
        return False

    def cast_ray(self, angle, grid, predators, targets):
        dx, dy = np.cos(angle), np.sin(angle)
        min_dist = 15.0
        obj_type = 0

        # Wall Cast
        for d in np.linspace(0, 15.0, 30):
            cx, cy = self.pos[0] + dx * d, self.pos[1] + dy * d
            if self.check_collision([cx, cy], grid):
                min_dist = float(d)
                obj_type = 1
                break

        # Predator Cast
        for predator_pos in predators:
            to_pred = predator_pos - self.pos
            dist_pred = np.linalg.norm(to_pred)
            if dist_pred < min_dist and dist_pred > 0:
                pred_angle = np.arctan2(to_pred[1], to_pred[0])
                if angdiff(angle, pred_angle) < 0.1:
                    min_dist = dist_pred
                    obj_type = 2
        
        # Target Cast
        for t in targets:
            to_t = t - self.pos
            dist_t = np.linalg.norm(to_t)
            if dist_t < min_dist and dist_t > 0:
                t_angle = np.arctan2(to_t[1], to_t[0])
                if angdiff(angle, t_angle) < 0.1:
                    min_dist = dist_t
                    obj_type = 3

        return min_dist, obj_type

    def update_sensors(self, dt, grid, predators, targets):
        self.whisk_phase += 0.8 * (dt / 0.05)
        self.whisk_angle = np.sin(self.whisk_phase) * (np.pi / 4.0)
        heading = np.arctan2(self.vel[1], self.vel[0]) if np.linalg.norm(self.vel) > 0 else self.heading
        self.whisker_hits = [False, False]

        angles = [(-np.pi / 4) + self.whisk_angle, (np.pi / 4) - self.whisk_angle]
        for i, ang in enumerate(angles):
            glob_ang = heading + ang
            tip = self.pos + np.array([np.cos(glob_ang), np.sin(glob_ang)]) * 5.0
            if self.check_collision(tip, grid):
                self.whisker_hits[i] = True

        self.vision_buffer = []
        fov = np.pi * (2 / 3)
        for i in range(12):
            offset = (i / 11.0 - 0.5) * fov
            ray_ang = heading + offset
            dist, obj = self.cast_ray(ray_ang, grid, predators, targets)
            self.vision_buffer.append({"dist": dist, "type": obj, "angle": ray_ang})

    def receive_reward(self):
        """Called by TournamentManager when this agent hits a target."""
        self.score += 1
        self.frustration = 0.0
        self.dopamine_tonic = min(0.5, self.dopamine_tonic + 0.1)
        return 1.0

    def step(self, dt, env_state, external_reward=0.0):
        grid = env_state["grid"]
        targets = env_state["targets"]
        predators = env_state["predators"]

        self.update_sensors(dt, grid, predators, targets)
        self.frames_alive += 1

        # Update Head Direction
        if np.linalg.norm(self.vel) > 0.01:
            new_heading = float(np.arctan2(self.vel[1], self.vel[0]))
            ang_vel = angdiff(new_heading, self.heading)
            self.heading = new_heading
            self.brain.update_hd_cells(ang_vel)
        hd = self.brain.get_head_direction()

        # Collision Check
        next_pos = self.pos + self.vel * (dt / 0.05)
        collision = self.check_collision(next_pos, grid)
        if collision:
            self.frustration = min(1.0, self.frustration + 0.1)
            self.vel *= -0.5

        min_dist_pred = 1000.0
        closest_predator = None
        for predator_pos in predators:
            dist = np.linalg.norm(self.pos - predator_pos)
            if dist < min_dist_pred:
                min_dist_pred = dist
                closest_predator = predator_pos
        
        danger = max(0.0, 1.0 - min_dist_pred / 10.0)

        # Run Brain
        soma, theta, grid_viz, decision, gl, atp, pain, touch, pm, sr, tr, mp, mg, idopa, iphasic = self.brain.process_votes(
            frustration=self.frustration,
            dopamine_tonic=self.dopamine_tonic,
            rat_vel=self.vel,
            reward_signal=external_reward,
            danger_level=danger,
            pheromones_len=0,
            head_direction=hd,
            vision_data=self.vision_buffer,
            anesthetic_level=0.0,
            fear_gain=self.config.get("fear_gain", 1.0),
            memory_gain=self.config.get("memory_gain", 1.0),
            energy_constraint=True,
            whisker_hits=self.whisker_hits,
            collision=collision,
            rat_pos=self.pos,
            dt=dt,
            predator_pos=closest_predator
        )

        # Physics
        self.vel = (self.vel * 0.85) + (decision * 0.15)
        speed = np.linalg.norm(self.vel)
        if speed > 0.6:
            self.vel = (self.vel / speed) * 0.6

        if not collision:
            self.pos += self.vel * (dt / 0.05)

        self.dopamine_phasic = idopa - self.dopamine_tonic
        self.brain_data = {"soma": soma, "theta": theta, "grid": grid_viz, "heading": float(self.heading)}

        return self.pos
