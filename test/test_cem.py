import grasp_opt
from matplotlib import pyplot as plt
import torch

def sphere(x):
    return torch.sum(torch.square(x), dim=-1)

def rosenbrock(x):
    return torch.sum(torch.square(1 - x[:, :-1]) + 100 * torch.square(x[:, 1:] - torch.square(x[:, :-1])), dim=-1)

# mu_0 = torch.randn(5)
mu_0 = torch.ones(5)
Sigma_0 = 1e0 * torch.eye(5)

mu_f, Sigma_f, cost_history, _ = grasp_opt.optimize_cem(rosenbrock, mu_0, Sigma_0, num_samples=1000, elite_frac=0.1)
print(mu_f)

plt.plot([torch.mean(cc).detach().cpu() for cc in cost_history])
plt.show()