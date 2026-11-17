import random
import numpy as np

class DynamicObstacles:
    def __init__(self, count=5, seed=None):
        self.rng=random.Random(seed); self.count=count; self.items=[]; self.timer=0.0
    def reset(self, base_grid, start, goal):
        self.items=[]; h,w=base_grid.shape
        reserved={start,goal}
        free=[(r,c) for r in range(2,h-2) for c in range(2,w-2)
              if not base_grid[r,c] and (r,c) not in reserved]
        self.rng.shuffle(free)
        for cell in free[:self.count]:
            self.items.append({'cell':cell,'wait':self.rng.uniform(0,0.5)})
        self.timer=0.0
    def update(self, base_grid, dt):
        self.timer += dt
        if self.timer >= 0.55:
            self.timer=0.0; h,w=base_grid.shape
            occupied={x['cell'] for x in self.items}
            for item in self.items:
                r,c=item['cell']; choices=[(r+1,c),(r-1,c),(r,c+1),(r,c-1)]
                self.rng.shuffle(choices)
                for nr,nc in choices:
                    if 1<=nr<h-1 and 1<=nc<w-1 and not base_grid[nr,nc] and (nr,nc) not in occupied:
                        occupied.discard(item['cell']); item['cell']=(nr,nc); occupied.add((nr,nc)); break
        grid=base_grid.copy()
        for r,c in (x['cell'] for x in self.items): grid[r,c]=1
        return grid
