"""The goal of this file to implement the diffusion process for the DexDiffuser.
   Implementation based on: https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py
"""
import os
from tqdm import tqdm
import torch.nn as nn
import time
import torch
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from nerf_grasping.dexdiffuser.dex_sampler import DexSampler
from nerf_grasping.dexdiffuser.diffusion_config import Config
from nerf_grasping.dexdiffuser.grasp_bps_dataset import GraspBPSSampleDataset, GraspBPSEvalDataset
from torch.utils.data import random_split


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_optimizer(config: Config, parameters):
    if config.optim.optimizer == "Adam":
        return optim.Adam(
            parameters,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(config.optim.beta1, 0.999),
            amsgrad=config.optim.amsgrad,
            eps=config.optim.eps,
        )
    elif config.optim.optimizer == "RMSProp":
        return optim.RMSprop(
            parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer == "SGD":
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            "Optimizer {} not understood.".format(config.optim.optimizer)
        )


def noise_estimation_loss(
    model,
    x0: torch.Tensor,
    cond: torch.Tensor,
    t: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    keepdim=False,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    output = model(f_O=cond, g_t=x, t=t.float().view(-1, 1))
    if keepdim:
        return (e - output).square().sum(dim=1)
    else:
        return (e - output).square().sum(dim=1).mean(dim=0)


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device
            )
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)
    return a


def generalized_steps(x, cond, seq, model, b, eta):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to("cuda")
            et = model(g_t=xt, f_O=cond, t=t.float().view(-1, 1))
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to("cpu"))

    return xs, x0_preds


def ddpm_steps(x, cond, seq, model, b):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to("cuda")

            output = model(g_t=x, f_O=cond, t=t.float().view(-1, 1))
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to("cpu"))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e
                + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to("cpu"))
    return xs, x0_preds


def get_dataset(
    hdf5_path: str | None = "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset.hdf5",
    use_evaluator_dataset: bool = False,
    get_all_labels: bool = False,
) -> tuple[
    GraspBPSSampleDataset | GraspBPSEvalDataset,
    GraspBPSSampleDataset | GraspBPSEvalDataset,
    GraspBPSSampleDataset | GraspBPSEvalDataset,
]:
    if use_evaluator_dataset:
        full_dataset = GraspBPSEvalDataset(
            input_hdf5_filepath=hdf5_path,
            get_all_labels=get_all_labels,
        )
    else:
        full_dataset = GraspBPSSampleDataset(
            input_hdf5_filepath=hdf5_path,
            get_all_labels=get_all_labels,
        )
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train_dataset, test_dataset, full_dataset


class Diffusion(object):
    def __init__(self, config: Config, device=None):
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # self.model = nn.DataParallel(
        #     DexSampler(
        #         n_pts=config.data.n_pts,
        #         grasp_dim=config.data.grasp_dim,
        #         d_model=128,
        #         virtual_seq_len=4,
        #     )
        # ).to(self.device)  # TODO(ahl): try to get this set up
        self.model = DexSampler(
            n_pts=config.data.n_pts,
            grasp_dim=config.data.grasp_dim,
            d_model=128,
            virtual_seq_len=4,
        ).to(self.device)

    def load_checkpoint(self, config: Config) -> None:
        states = torch.load(
            config.training.log_path / "ckpt.pth",
            map_location=self.device,
        )
        model_state_dict = states[0]
        self.model.load_state_dict(model_state_dict)

    def train(self) -> None:
        config = self.config
        train_dataset, test_dataset, _ = get_dataset(get_all_labels=False)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        optimizer = get_optimizer(self.config, self.model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0

        for epoch in tqdm(
            range(start_epoch, self.config.training.n_epochs), desc="Training Epochs"
        ):
            data_start = time.time()
            data_time = 0
            for i, (grasps, bpss, _) in tqdm(
                enumerate(train_loader),
                desc="Training Batches",
                total=len(train_loader),
            ):
                time.sleep(0.1)  # Yield control so it can be interrupted
                n = grasps.size(0)
                data_time += time.time() - data_start
                self.model.train()
                step += 1

                grasps = grasps.to(self.device)
                bpss = bpss.to(self.device)
                e = torch.randn_like(grasps)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = noise_estimation_loss(
                    model=self.model, x0=grasps, cond=bpss, t=t, e=e, b=b
                )

                if step % config.training.print_freq == 0 or step == 1:
                    print(
                        f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                    )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(self.model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        self.model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    log_path = config.training.log_path
                    log_path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        states,
                        log_path / "ckpt_{}.pth".format(step),
                    )
                    torch.save(states, log_path / "ckpt.pth")

                data_start = time.time()

    def sample(self, xT: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        with torch.no_grad():
            x = self.sample_image(xT, cond, self.model)
            return x

    def sample_image(
        self,
        x,
        cond,
        model,
        last=True,
        sample_type="generalized",
        skip_type="uniform",
        skip=1,
        timesteps=1000,
        eta=0.0,
    ) -> torch.Tensor:
        if skip_type == "uniform":
            skip = self.num_timesteps // timesteps
            seq = range(0, self.num_timesteps, skip)
        elif skip_type == "quad":
            seq = np.linspace(0, np.sqrt(self.num_timesteps * 0.8), timesteps) ** 2
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        if sample_type == "generalized":
            xs = generalized_steps(x, cond, seq, model, self.betas, eta=eta)
            x = xs
        elif sample_type == "ddpm_noisy":
            x = ddpm_steps(x, cond, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x


def main() -> None:
    config = Config()
    runner = Diffusion(config)

    TRAIN_MODE = True
    if TRAIN_MODE:
        runner.train()
    else:
        runner.load_checkpoint(config)

        _, _, full_dataset = get_dataset(get_all_labels=False)
        GRASP_IDX = 13021
        _, bps, _ = full_dataset[GRASP_IDX]
        xT = torch.randn(1, config.data.grasp_dim, device=runner.device)
        x = runner.sample(xT=xT, cond=bps[None].to(runner.device)).squeeze().cpu()
        print(f"Sampled grasp shape: {x.shape}")

        import numpy as np
        import trimesh
        import pathlib
        import transforms3d
        import open3d as o3d
        import plotly.graph_objects as go
        from nerf_grasping.dexgraspnet_utils.hand_model import HandModel
        from nerf_grasping.dexgraspnet_utils.hand_model_type import (
            HandModelType,
        )
        from nerf_grasping.dexgraspnet_utils.pose_conversion import (
            hand_config_to_pose,
        )
        from nerf_grasping.dexgraspnet_utils.joint_angle_targets import (
            compute_optimized_joint_angle_targets_given_grasp_orientations,
        )

        MESHDATA_ROOT = (
            "/home/albert/research/nerf_grasping/rsync_meshes/rotated_meshdata_v2"
        )
        print("=" * 79)
        print(f"len(full_dataset): {len(full_dataset)}")

        print("\n" + "=" * 79)
        print(f"Getting grasp and bps for grasp_idx {GRASP_IDX}")
        print("=" * 79)
        grasp = x
        passed_eval = np.array(1)
        # grasp, bps, passed_eval = full_dataset[GRASP_IDX]
        print(f"grasp.shape: {grasp.shape}")
        print(f"bps.shape: {bps.shape}")
        print(f"passed_eval.shape: {passed_eval.shape}")

        print("\n" + "=" * 79)
        print("Getting debugging extras")
        print("=" * 79)
        basis_points = full_dataset.get_basis_points()
        object_code = full_dataset.get_object_code(GRASP_IDX)
        object_scale = full_dataset.get_object_scale(GRASP_IDX)
        object_state = full_dataset.get_object_state(GRASP_IDX)
        print(f"basis_points.shape: {basis_points.shape}")

        # Mesh
        mesh_path = pathlib.Path(f"{MESHDATA_ROOT}/{object_code}/coacd/decomposed.obj")
        assert mesh_path.exists(), f"{mesh_path} does not exist"
        print(f"Reading mesh from {mesh_path}")
        mesh = trimesh.load(mesh_path)

        xyz, quat_xyzw = object_state[:3], object_state[3:7]
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        transform = np.eye(4)  # X_W_Oy
        transform[:3, :3] = transforms3d.quaternions.quat2mat(quat_wxyz)
        transform[:3, 3] = xyz
        mesh.apply_scale(object_scale)
        mesh.apply_transform(transform)

        # Point cloud
        point_cloud_filepath = full_dataset.get_point_cloud_filepath(GRASP_IDX)
        print(f"Reading point cloud from {point_cloud_filepath}")
        point_cloud = o3d.io.read_point_cloud(point_cloud_filepath)
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
        point_cloud_points = np.asarray(point_cloud.points)
        print(f"point_cloud_points.shape: {point_cloud_points.shape}")

        # Grasp
        assert grasp.shape == (
            3 + 6 + 16 + 4 * 3,
        ), f"Expected shape (3 + 6 + 16 + 4 * 3), got {grasp.shape}"
        grasp = grasp.detach().cpu().numpy()
        grasp_trans, grasp_rot6d, grasp_joints, grasp_dirs = (
            grasp[:3],
            grasp[3:9],
            grasp[9:25],
            grasp[25:].reshape(4, 3),
        )
        grasp_rot = np.zeros((3, 3))
        grasp_rot[:3, :2] = grasp_rot6d.reshape(3, 2)
        grasp_rot[:3, 0] = grasp_rot[:3, 0] / np.linalg.norm(grasp_rot[:3, 0])
        # make grasp_rot[:3, 1] orthogonal to grasp_rot[:3, 0]
        grasp_rot[:3, 1] = (
            grasp_rot[:3, 1]
            - np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) * grasp_rot[:3, 0]
        )
        grasp_rot[:3, 1] = grasp_rot[:3, 1] / np.linalg.norm(grasp_rot[:3, 1])
        assert (
            np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) < 1e-3
        ), f"Expected dot product < 1e-3, got {np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1])}"
        grasp_rot[:3, 2] = np.cross(grasp_rot[:3, 0], grasp_rot[:3, 1])
        grasp_transform = np.eye(4)  # X_Oy_H
        grasp_transform[:3, :3] = grasp_rot
        grasp_transform[:3, 3] = grasp_trans
        print(f"grasp_transform:\n{grasp_transform}")
        grasp_transform = transform @ grasp_transform  # X_W_H = X_W_Oy @ X_Oy_H
        grasp_trans = grasp_transform[:3, 3]
        grasp_rot = grasp_transform[:3, :3]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hand_pose = hand_config_to_pose(
            grasp_trans[None], grasp_rot[None], grasp_joints[None]
        ).to(device)
        hand_model_type = HandModelType.ALLEGRO_HAND
        grasp_orientations = np.zeros(
            (4, 3, 3)
        )  # NOTE: should have applied transform with this, but didn't because we only have z-dir, hopefully transforms[:3, :3] ~= np.eye(3)
        grasp_orientations[:, :, 2] = (
            grasp_dirs  # Leave the x-axis and y-axis as zeros, hacky but works
        )
        hand_model = HandModel(hand_model_type=hand_model_type, device=device)
        hand_model.set_parameters(hand_pose)
        hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.8)

        (
            optimized_joint_angle_targets,
            _,
        ) = compute_optimized_joint_angle_targets_given_grasp_orientations(
            joint_angles_start=hand_model.hand_pose[:, 9:],
            hand_model=hand_model,
            grasp_orientations=torch.from_numpy(grasp_orientations[None]).to(device),
        )
        new_hand_pose = hand_config_to_pose(
            grasp_trans[None],
            grasp_rot[None],
            optimized_joint_angle_targets.detach().cpu().numpy(),
        ).to(device)
        hand_model.set_parameters(new_hand_pose)
        hand_plotly_optimized = hand_model.get_plotly_data(
            i=0, opacity=0.3, color="lightgreen"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=basis_points[:, 0],
                y=basis_points[:, 1],
                z=basis_points[:, 2],
                mode="markers",
                marker=dict(
                    size=1,
                    color=bps,
                    colorscale="rainbow",
                    colorbar=dict(title="Basis points", orientation="h"),
                ),
                name="Basis points",
            )
        )
        fig.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                name="Object",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=point_cloud_points[:, 0],
                y=point_cloud_points[:, 1],
                z=point_cloud_points[:, 2],
                mode="markers",
                marker=dict(size=1.5, color="black"),
                name="Point cloud",
            )
        )
        fig.update_layout(
            title=dict(
                text=f"Grasp idx: {GRASP_IDX}, Object: {object_code}, Passed Eval: {passed_eval}"
            ),
        )
        VISUALIZE_HAND = True
        if VISUALIZE_HAND:
            for trace in hand_plotly:
                fig.add_trace(trace)
            for trace in hand_plotly_optimized:
                fig.add_trace(trace)
        fig.show()


if __name__ == "__main__":
    main()
