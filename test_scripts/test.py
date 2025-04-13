import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
observation_space = (12288,)
a = torch.randn(1, 12288).view(-1, *observation_space).permute(0, 3, 1, 2)

print(a.shape)