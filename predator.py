import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class Predator:
    def __init__(self, pos, direction="horizontal"):
        self.pos = np.array(pos)
        self.vel = np.array([0.0, 0.0])
        # Slightly faster now to force the rat to be smart
        self.speed = 0.24
        self.direction = direction
        self.patrol_direction = 1

    def patrol_wall(self, map_size, dt_scale: float = 1.0):
        if self.direction == "horizontal":
            self.pos[0] += self.speed * self.patrol_direction * dt_scale
            if self.pos[0] >= map_size - 1 or self.pos[0] <= 0:
                self.patrol_direction *= -1
        else: # vertical
            self.pos[1] += self.speed * self.patrol_direction * dt_scale
            if self.pos[1] >= map_size - 1 or self.pos[1] <= 0:
                self.patrol_direction *= -1

    def hunt(self, rat_pos, rat_vel, grid, map_size, dt_scale: float = 1.0):
        # 1. Calculate Intercept
        dist = float(np.linalg.norm(rat_pos - self.pos))
        
        # Simple "Time to Impact" calculation
        # If we are far, look further ahead. If close, look at the rat.
        lookahead = clamp(dist / self.speed, 0.0, 15.0)
        
        # Predicted position of the rat (Target Leading)
        predicted_pos = rat_pos + (rat_vel * lookahead)
        
        # 2. Vector Navigation
        diff = predicted_pos - self.pos
        dist_pred = np.linalg.norm(diff)
        
        if dist_pred > 0:
            desired_vel = (diff / dist_pred) * (self.speed * dt_scale)
            next_pos = self.pos + desired_vel
            
            # Simple Wall Collision Check
            nx, ny = int(next_pos[0]), int(next_pos[1])
            if 0 <= nx < map_size and 0 <= ny < map_size:
                if grid[ny, nx] == 0:
                    self.pos = next_pos
                else:
                    # Slide along wall
                    if grid[ny, int(self.pos[0])] == 0:
                        self.pos[1] = next_pos[1]
                    elif grid[int(self.pos[1]), nx] == 0:
                        self.pos[0] = next_pos[0]
