import torch
import torch.nn as nn

class FighterPolicyNet(nn.Module):
    def __init__(self, obs_dim=17, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, x):
        return self.net(x)

# Load and inspect
state = torch.load("fighter_policy.pt", map_location="cpu")
print("Model keys:", state.keys())
print("First layer shape:", state['net.0.weight'].shape)
print("Expected shape: torch.Size([256, 17])")

# Try to load into 17D model
model = FighterPolicyNet(obs_dim=17)
model.load_state_dict(state)
print("✓ Model loaded successfully with 17D inputs!")