import torch
import grasp_opt


N = 1000
n_z = torch.randn(N, 3)
n_z = n_z / torch.norm(n_z, dim=-1, keepdim=True)
R = grasp_opt.rot_from_vec(n_z)

e3 = torch.tensor([0,0,1]).reshape(1,3, 1).expand(N, 3, 1).float()

print(torch.mean(torch.norm((R@e3).squeeze() - n_z)))