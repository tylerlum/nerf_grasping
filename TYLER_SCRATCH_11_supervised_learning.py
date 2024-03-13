# %%
import torch
import pathlib
import numpy as np

# from tqdm import tqdm
from tqdm.notebook import tqdm
from localscope import localscope
from typing import List, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


# %%
@dataclass
class Config:
    grasp_config_dicts_folder: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-03-10_75bottle_augmented_pose_HALTON_400/evaled_grasp_config_dicts_train/"
    )
    batch_size: int = 256
    n_epochs: int = 100
    hidden_size: List[int] = field(
        default_factory=lambda: [512, 512]
    )


# %%
cfg = Config()
assert cfg.grasp_config_dicts_folder.exists()
grasp_config_dict_paths = sorted(list(cfg.grasp_config_dicts_folder.glob("*.npy")))
assert len(grasp_config_dict_paths) > 0
print(f"len(grasp_config_dict_paths): {len(grasp_config_dict_paths)}")

# %%
N_OBJECTS = len(grasp_config_dict_paths)
grasp_config_dicts = [
    np.load(grasp_config_dict_path, allow_pickle=True).item()
    for grasp_config_dict_path in grasp_config_dict_paths
]
n_grasps_per_object_list = [
    len(gc_dict["passed_eval"]) for gc_dict in grasp_config_dicts
]
n_grasps_per_object = n_grasps_per_object_list[0]
assert all(n_grasps_per_object == n_grasps for n_grasps in n_grasps_per_object_list)
n_grasps = n_grasps_per_object * N_OBJECTS

# %%
trans = np.stack([gc_dict["trans"] for gc_dict in grasp_config_dicts], axis=0)
rot = np.stack([gc_dict["rot"] for gc_dict in grasp_config_dicts], axis=0)
joint_angles = np.stack(
    [gc_dict["joint_angles"] for gc_dict in grasp_config_dicts], axis=0
)
grasp_orientations = np.stack(
    [gc_dict["grasp_orientations"] for gc_dict in grasp_config_dicts], axis=0
)

object_ids = np.stack(
    [np.zeros(n_grasps_per_object) + i for i in range(N_OBJECTS)], axis=0
)
passed_evals = np.stack(
    [gc_dict["passed_eval"] for gc_dict in grasp_config_dicts], axis=0
)
ROUND_LABELS = True
if ROUND_LABELS:
    print(f"Rounding labels to 0 or 1")
    passed_evals = (passed_evals > 0.5).astype(int)

assert trans.shape == (N_OBJECTS, n_grasps_per_object, 3)
assert rot.shape == (N_OBJECTS, n_grasps_per_object, 3, 3)
assert joint_angles.shape == (N_OBJECTS, n_grasps_per_object, 16)
assert grasp_orientations.shape == (N_OBJECTS, n_grasps_per_object, 4, 3, 3)
assert object_ids.shape == (N_OBJECTS, n_grasps_per_object)
assert passed_evals.shape == (N_OBJECTS, n_grasps_per_object)

# %%
print(f"N_OBJECTS: {N_OBJECTS}")
print(f"n_grasps_per_object: {n_grasps_per_object}")
print(f"n_grasps: {n_grasps}")


# %%
class GraspDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trans: np.ndarray,
        rot: np.ndarray,
        joint_angles: np.ndarray,
        grasp_orientations: np.ndarray,
        object_ids: np.ndarray,
        passed_evals: np.ndarray,
    ) -> None:
        super().__init__()
        self.trans = torch.from_numpy(trans).float()
        self.rot = torch.from_numpy(rot).float()
        self.joint_angles = torch.from_numpy(joint_angles).float()
        self.grasp_orientations = torch.from_numpy(grasp_orientations).float()
        self.object_ids = torch.from_numpy(object_ids).long()
        self.passed_evals = torch.from_numpy(passed_evals).float()

        N = len(self)
        assert trans.shape == (N, 3)
        assert rot.shape == (N, 3, 3)
        assert joint_angles.shape == (N, 16)
        assert grasp_orientations.shape == (N, 4, 3, 3)
        assert object_ids.shape == (N,)
        assert passed_evals.shape == (N,)

    def __len__(self) -> int:
        return self.trans.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.trans[idx],
            self.rot[idx],
            self.joint_angles[idx],
            self.grasp_orientations[idx],
            self.object_ids[idx],
            self.passed_evals[idx],
        )


all_grasp_dataset = GraspDataset(
    trans=trans.reshape(-1, 3),
    rot=rot.reshape(-1, 3, 3),
    joint_angles=joint_angles.reshape(-1, 16),
    grasp_orientations=grasp_orientations.reshape(-1, 4, 3, 3),
    object_ids=object_ids.reshape(-1),
    passed_evals=passed_evals.reshape(-1),
)

# %%
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    all_grasp_dataset,
    [0.8, 0.1, 0.1],
    generator=torch.Generator().manual_seed(42),
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg.batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=cfg.batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=cfg.batch_size, shuffle=False
)


# %%
class GraspModel(torch.nn.Module):
    def __init__(self, hidden_size: List[int]) -> None:
        super().__init__()
        input_size = 3 + 9 + 16 + 36 + N_OBJECTS
        output_size = 2
        self.fc_list = torch.nn.ModuleList()
        sizes = [input_size] + hidden_size
        for size1, size2 in zip(sizes[:-1], sizes[1:]):
            self.fc_list.append(torch.nn.Linear(size1, size2))

        self.fc_out = torch.nn.Linear(hidden_size[-1], output_size)

    def forward(
        self,
        trans: torch.Tensor,
        rot: torch.Tensor,
        joint_angles: torch.Tensor,
        grasp_orientations: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = trans.shape[0]
        object_ids_onehot = torch.nn.functional.one_hot(
            input=object_ids, num_classes=N_OBJECTS
        ).float()

        x = torch.cat(
            [
                trans.reshape(batch_size, -1),
                rot.reshape(batch_size, -1),
                joint_angles.reshape(batch_size, -1),
                grasp_orientations.reshape(batch_size, -1),
                object_ids_onehot.reshape(batch_size, -1),
            ],
            dim=1,
        )
        assert x.shape == (batch_size, 3 + 9 + 16 + 36 + N_OBJECTS)

        for i, fc in enumerate(self.fc_list):
            x = torch.nn.functional.relu(fc(x))
        return self.fc_out(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GraspModel(hidden_size=cfg.hidden_size).to(device)
model.train()

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_losses_per_epoch, val_losses_per_epoch = [], []

# %%
# Compute class weight
from sklearn.utils.class_weight import compute_class_weight
passed_evals_rounded = (passed_evals.reshape(-1) > 0.01).astype(int)

class_weight = compute_class_weight(
    class_weight="balanced", classes=np.unique(passed_evals_rounded), y=passed_evals_rounded
)
assert class_weight.shape == (2,)

# %%
ce_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(class_weight).float().to(device))

# %%
class WeightedL2Loss(torch.nn.Module):
    def __init__(self, zero_weight: float, nonzero_weight: float) -> None:
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape
        assert input.shape[1] == target.shape[1] == 2

        weight = torch.where(
            target[:, 0] > 0.99, 
            self.zero_weight * torch.ones(len(target)).to(device),
            self.nonzero_weight * torch.ones(len(target)).to(device),
        )

        raw_loss = torch.nn.functional.mse_loss(torch.nn.functional.softmax(input, dim=1), target, reduction="none")
        assert raw_loss.shape == (len(target), 2)
        loss_per_sample = torch.sum(raw_loss, dim=1)
        assert loss_per_sample.shape == (len(target),)

        weighted_loss_per_sample = weight * loss_per_sample

        return torch.mean(weighted_loss_per_sample)

weighted_l2_loss_fn = WeightedL2Loss(zero_weight=class_weight[0], nonzero_weight=class_weight[1])
# loss_fn = weighted_l2_loss_fn
loss_fn = ce_loss_fn

# %%
@localscope.mfc
def train(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
) -> List[float]:
    train_losses = []
    n_batches = len(loader)
    pbar = tqdm(loader, total=n_batches, desc="Training")
    for trans, rot, joint_angles, grasp_orientations, object_ids, passed_evals in pbar:
        optimizer.zero_grad()

        trans = trans.to(device)
        rot = rot.to(device)
        joint_angles = joint_angles.to(device)
        grasp_orientations = grasp_orientations.to(device)
        object_ids = object_ids.to(device)
        passed_evals = passed_evals.to(device)

        pred = model(
            trans=trans,
            rot=rot,
            joint_angles=joint_angles,
            grasp_orientations=grasp_orientations,
            object_ids=object_ids,
        )
        assert pred.shape == (len(passed_evals), 2)
        assert passed_evals.shape == (len(passed_evals),)
        assert torch.all(0 <= passed_evals) and torch.all(passed_evals <= 1)
        passed_evals = torch.stack([1 - passed_evals, passed_evals], dim=1)
        assert passed_evals.shape == pred.shape
        assert torch.allclose(
            torch.sum(passed_evals, dim=1), torch.ones(len(passed_evals)).to(device)
        )
        loss = loss_fn(input=pred, target=passed_evals)
        loss.backward()
        train_losses.append(loss.item())

        optimizer.step()
        pbar.set_description(f"Training, loss: {np.mean(train_losses):.4f}")
    return train_losses


@localscope.mfc
def val(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: str,
) -> List[float]:
    val_losses = []
    n_batches = len(loader)
    for trans, rot, joint_angles, grasp_orientations, object_ids, passed_evals in tqdm(
        loader, total=n_batches, desc="Validation"
    ):

        trans = trans.to(device)
        rot = rot.to(device)
        joint_angles = joint_angles.to(device)
        grasp_orientations = grasp_orientations.to(device)
        object_ids = object_ids.to(device)
        passed_evals = passed_evals.to(device)

        with torch.no_grad():
            pred = model(
                trans=trans,
                rot=rot,
                joint_angles=joint_angles,
                grasp_orientations=grasp_orientations,
                object_ids=object_ids,
            )
            assert pred.shape == (len(passed_evals), 2)
            assert passed_evals.shape == (len(passed_evals),)
            assert torch.all(0 <= passed_evals) and torch.all(passed_evals <= 1)
            passed_evals = torch.stack([1 - passed_evals, passed_evals], dim=1)
            assert passed_evals.shape == pred.shape
            assert torch.allclose(
                torch.sum(passed_evals, dim=1), torch.ones(len(passed_evals)).to(device)
            )
            loss = loss_fn(input=pred, target=passed_evals)
            val_losses.append(loss.item())

    return val_losses

# %%
@localscope.mfc
def get_prediction_and_truths(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n_batches = len(loader)
    preds, truths = [], []
    for trans, rot, joint_angles, grasp_orientations, object_ids, passed_evals in tqdm(
        loader, total=n_batches, desc="Forward pass"
    ):

        trans = trans.to(device)
        rot = rot.to(device)
        joint_angles = joint_angles.to(device)
        grasp_orientations = grasp_orientations.to(device)
        object_ids = object_ids.to(device)
        passed_evals = passed_evals.to(device)

        with torch.no_grad():
            pred = model(
                trans=trans,
                rot=rot,
                joint_angles=joint_angles,
                grasp_orientations=grasp_orientations,
                object_ids=object_ids,
            )
            assert pred.shape == (len(passed_evals), 2)
            assert passed_evals.shape == (len(passed_evals),)
            assert torch.all(0 <= passed_evals) and torch.all(passed_evals <= 1)
            passed_evals = torch.stack([1 - passed_evals, passed_evals], dim=1)
            assert passed_evals.shape == pred.shape
            assert torch.allclose(
                torch.sum(passed_evals, dim=1), torch.ones(len(passed_evals)).to(device)
            )
            truths += passed_evals[:, 1].detach().cpu().numpy().tolist()
            preds += torch.nn.functional.softmax(pred, dim=1)[:, 1].detach().cpu().numpy().tolist()

    return np.array(preds), np.array(truths)
# %%
@localscope.mfc
def _create_histogram(
    ground_truths: List[float],
    predictions: List[float],
    title: str,
    match_ylims: bool = True,
) -> plt.Figure:
    unique_labels = np.unique(ground_truths)
    fig, axes = plt.subplots(len(unique_labels), 1, figsize=(10, 20))
    axes = axes.flatten()

    # Get predictions per label
    unique_labels_to_preds = {}
    for i, unique_label in enumerate(unique_labels):
        preds = np.array(predictions)
        idxs = np.array(ground_truths) == unique_label
        unique_labels_to_preds[unique_label] = preds[idxs]

    # Plot histogram per label
    for i, (unique_label, preds) in enumerate(
        sorted(unique_labels_to_preds.items())
    ):
        axes[i].hist(preds, bins=50, alpha=0.7, color="blue")
        axes[i].plot([unique_label, unique_label], axes[i].get_ylim(), c="r", label="Ground Truth")
        axes[i].set_title(f"Ground Truth: {unique_label}")
        axes[i].set_xlim(0, 1)
        axes[i].set_xlabel("Prediction")
        axes[i].set_ylabel("Count")

    # Matching ylims
    if match_ylims:
        max_y_val = max(ax.get_ylim()[1] for ax in axes)
        for i in range(len(axes)):
            axes[i].set_ylim(0, max_y_val)

    fig.suptitle(title)
    fig.tight_layout()
    return fig





# %%
for epoch in tqdm(range(cfg.n_epochs), desc="Epoch"):
    # Train
    train_losses = train(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    train_losses_per_epoch.append(np.mean(train_losses))

    # Val
    val_losses = val(loader=val_loader, model=model, loss_fn=loss_fn, device=device)
    val_losses_per_epoch.append(np.mean(val_losses))

    # Plot
    if epoch % 5 == 0:
        preds, truths = get_prediction_and_truths(loader=val_loader, model=model, device=device)
    fig = _create_histogram(ground_truths=truths, predictions=preds, title="Grasp Metric", match_ylims=False)

    print(f"{epoch=}, {train_losses_per_epoch[-1]=}, {val_losses_per_epoch[-1]=}")


# %%
plt.plot(train_losses_per_epoch, label="Train")
plt.plot(val_losses_per_epoch, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
preds, truths = get_prediction_and_truths(loader=train_loader, model=model, device=device)
# preds, truths = get_prediction_and_truths(loader=train_loader, model=model, device=device)

# %%
noise = np.random.normal(0, 0.02, len(truths))
plt.scatter(truths + noise, preds + noise, s=1)

# %%
fig = _create_histogram(ground_truths=truths, predictions=preds, title="Grasp Metric", match_ylims=False)


# %%
# Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(truths > 0.5, preds > 0.5)
print(cm)


# %%
# Confusion matrix plot
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(cmap="Blues")


# %%
# Confusion matrix plot with percentages
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(cmap="Blues", values_format=".2%")


# %%
