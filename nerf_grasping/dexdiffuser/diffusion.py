# %%
"""The goal of this file to implement the diffusion process for the DexDiffuser.
   Implementation based on: https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py
"""

# %%
from torchvision.datasets import CIFAR10
import time
import torch
import numpy as np
import yaml
import argparse
import torch.optim as optim
import torch.utils.data as data
import os


# %%
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


config = dict2namespace(
    yaml.safe_load(
        """data:
    dataset: "CIFAR10"
    image_size: 32
    channels: 3
    logit_transform: false
    uniform_dequantization: false
    gaussian_dequantization: false
    random_flip: true
    rescaled: true
    num_workers: 4

model:
    type: "simple"
    in_channels: 3
    out_ch: 3
    ch: 128
    ch_mult: [1, 2, 2, 2]
    num_res_blocks: 2
    attn_resolutions: [16, ]
    dropout: 0.1
    var_type: fixedlarge
    ema_rate: 0.9999
    ema: True
    resamp_with_conv: True

diffusion:
    beta_schedule: linear
    beta_start: 0.0001
    beta_end: 0.02
    num_diffusion_timesteps: 1000

training:
    batch_size: 128
    n_epochs: 10000
    n_iters: 5000000
    snapshot_freq: 5000
    validation_freq: 2000

sampling:
    batch_size: 64
    last_only: True

optim:
    weight_decay: 0.000
    optimizer: "Adam"
    lr: 0.0002
    beta1: 0.9
    amsgrad: false
    eps: 0.00000001
    grad_clip: 1.0
"""
    )
)

# %%
config


# %%
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


def get_optimizer(config, parameters):
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
    t: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    keepdim=False,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def get_dataset(config):
    dataset = CIFAR10(
        os.path.join("datasets", "cifar10"),
        train=True,
        download=True,
    )
    test_dataset = CIFAR10(
        os.path.join("datasets", "cifar10_test"),
        train=False,
        download=True,
    )
    return dataset, test_dataset



# %%
class Diffusion(object):
    def __init__(self, config, device=None):
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
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        config = self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = noise_estimation_loss(model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                print(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    log_path = os.path.join("exp", "logs", "tyler_test")
                    torch.save(
                        states,
                        os.path.join(log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(log_path, "ckpt.pth"))

                data_start = time.time()
