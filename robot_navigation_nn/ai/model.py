import torch
from torch import nn

class DQN(nn.Module):
    def __init__(self, input_size=27, hidden=(256, 128), output_size=5):
        super().__init__()
        layers=[]; last=input_size
        for width in hidden:
            layers += [nn.Linear(last, width), nn.ReLU()]
            last=width
        layers.append(nn.Linear(last, output_size))
        self.network=nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

# Kept for users who still have an old behavioral-cloning checkpoint.
class NavigationMLP(nn.Module):
    def __init__(self, input_size=27, hidden=(128,64,32), output_size=5):
        super().__init__(); layers=[]; last=input_size
        for width in hidden:
            layers += [nn.Linear(last,width), nn.ReLU()]; last=width
        layers.append(nn.Linear(last,output_size)); self.network=nn.Sequential(*layers)
    def forward(self,x): return self.network(x)

def load_policy(path, device="cpu"):
    model=DQN()
    state=torch.load(path,map_location=device,weights_only=True)
    model.load_state_dict(state); model.to(device).eval()
    return model
