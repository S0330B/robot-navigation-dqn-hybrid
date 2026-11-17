from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=50000): self.data=deque(maxlen=capacity)
    def add(self, state, action, reward, next_state, done):
        self.data.append((np.asarray(state,dtype=np.float32),int(action),float(reward),np.asarray(next_state,dtype=np.float32),bool(done)))
    def sample(self, batch_size):
        batch=random.sample(self.data,batch_size)
        s,a,r,ns,d=zip(*batch)
        return (np.stack(s),np.asarray(a,dtype=np.int64),np.asarray(r,dtype=np.float32),np.stack(ns),np.asarray(d,dtype=np.float32))
    def __len__(self): return len(self.data)
    def state_dict(self): return list(self.data)
    def load_state_dict(self, items): self.data=deque(items,maxlen=self.data.maxlen)
