import matplotlib.pyplot as plt

import grasp_utils


def plot_grasp_distribution(
    mu_f, coarse_model, fine_model, renderer, residual_dirs=False
):
    # Test grasp sampling

    rays, weights, z_vals = grasp_utils.get_grasp_distribution(
        mu_f.reshape(1, 3, 6),
        coarse_model,
        fine_model,
        renderer,
        residual_dirs=residual_dirs,
    )
    plt.close("all")
    for ii in range(3):
        plt.plot(
            z_vals[0, ii, :].detach().cpu().numpy().T,
            weights[0, ii, :].cpu().detach().numpy().T,
            label="Finger " + str(ii + 1),
        )
    plt.ylim([0, 0.2])
    plt.title("Grasp Point Distribution")
    plt.xlabel("Distance to Surface [m]")
    plt.ylabel("Probability Mass")
    plt.legend()
    plt.show()


def img_dir_to_vid(image_dir, name="test", cleanup=False):
    import glob
    import os

    import imageio

    writer = imageio.get_writer(f"{image_dir}/{name}.mp4", fps=20)
    img_files = sorted(
        glob.glob(os.path.join(image_dir, "img*.png")),
        key=lambda x: int(x.split("img")[1].split(".")[0]),
    )
    for file in img_files:
        im = imageio.imread(file)
        writer.append_data(im)
    writer.close()
    if cleanup:
        print("removing files")
        for file in img_files:
            os.remove(file)
