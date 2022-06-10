import logging
import os
import pdb
import time
import numpy as np

import sim_trifinger as sim
from nerf_grasping.sim.ig_viz_utils import img_dir_to_vid
from nerf_grasping.sim import ig_objects
import torch


def run_robot_control(tf, height, eplen=300, debug=False, save_dir=None):
    count = 0  # skip off step
    save_freq = 1
    while not tf.gym.query_viewer_has_closed(tf.viewer) and count < eplen:
        count += 1
        if save_dir:
            tf.save_viewer_frame(save_dir, save_freq)
        try:
            tf.step_gym()
            tf.robot.control(count, tf.object)
        except AssertionError:
            print("force optimization failed")
            # if debug:
            #     pdb.set_trace()
            return False
        except KeyboardInterrupt:
            pdb.set_trace()
        except Exception as e:
            raise e
        obj_height = tf.object.rb_states[0, 2]
    print(f"final height: {obj_height.numpy().item()}")
    logging.info(f"final height: {obj_height.numpy().item()}")
    return height < obj_height


def main(
    n_runs,
    height=0.07,
    obj="banana",
    metric="l1",
    eplen=250,
    viewer=True,
    debug=False,
    save_dir=None,
    cem_iters=10,
    nerf_grasping=True,
    use_grad_est=False,
    robot_type="trifinger",
    use_true_normals=False,
    grasp_data=None,
    grasp_idx=None,
):
    if obj == "banana":
        obj = ig_objects.Banana
    elif obj == "box":
        obj = ig_objects.Box
    elif obj == "teddy_bear":
        obj = ig_objects.TeddyBear
        obj.use_centroid = True
    elif obj == "powerdrill":
        obj = ig_objects.PowerDrill
    success = False
    succ_total = 0
    if robot_type == "trifinger":
        robot_kwargs = dict(
            metric=metric,
            cem_iters=cem_iters,
            use_grad_est=use_grad_est,
            use_true_normals=use_true_normals,
        )
    elif robot_type == "spheres":
        robot_kwargs = dict(use_grad_est=use_grad_est)

    tf = sim.TriFingerEnv(
        viewer=viewer,
        robot_type=robot_type,
        Obj=obj,
        use_nerf_grasping=nerf_grasping,
        use_residual_dirs=True,
        target_height=height,
        **robot_kwargs,
    )

    if grasp_data is not None:
        grasps = np.load(grasp_data)
        if grasp_idx is not None:
            n_runs = len(grasps)
        else:
            grasps = np.stack([grasps[grasp_idx] for _ in range(n_runs)])

    for i in range(n_runs):
        if save_dir:
            save_path = f"runs/{save_dir}/{i}"
        else:
            save_path = None
        if grasp_data is not None:
            grasp_points, grasp_normals = grasps[i, :, :3], grasps[i, :, 3:]
            grasp_vars = torch.tensor(grasp_points), torch.tensor(grasp_normals)
        else:
            grasp_vars = None
        tf.reset(grasp_vars=grasp_vars)
        success = run_robot_control(
            tf=tf,
            height=height,
            eplen=eplen,
            debug=debug,
            save_dir=save_path,
        )
        succ_total += success
        if save_dir:
            if success:
                print("success!")
                run_pre = "success"
            else:
                run_pre = "run"
            print(f"Saving vid to {save_path}/{run_pre}{i}.mp4")
            img_dir_to_vid(save_path, f"{run_pre}{i}", cleanup=not success)
    print("Pct success: {}%".format(succ_total / n_runs * 100))
    logging.info("Pct success: {}%".format(succ_total / n_runs * 100))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # run args
    parser.add_argument("--n", help="number of runs to evaluate", default=10, type=int)
    parser.add_argument(
        "--eplen", help="length of episodes for each run", default=300, type=int
    )
    parser.add_argument("--obj", "--o", help="object to use", default="banana")
    parser.add_argument(
        "--height", help="goal height to lift to", default=0.07, type=float
    )
    parser.add_argument(
        "--robot_type", default="spheres", choices=["trifinger", "spheres", ""]
    )
    parser.add_argument("--grasp_data", type=str, help="path to stored grasps")
    # debug args
    parser.add_argument("--debug", "--d", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--save_dir", "--s", default=None)
    # nerf args
    parser.add_argument("--no_nerf", action="store_true")
    parser.add_argument(
        "--metric", "--m", help="grasping metric to eval with", default="l1"
    )
    parser.add_argument("--cem_iters", default=10)
    parser.add_argument("--use_true_normals", action="store_true")
    parser.add_argument("--use_grad_est", action="store_true")
    parser.add_argument("--grasp_idx", default=None, type=int)

    args = parser.parse_args()
    handlers = [logging.StreamHandler()]
    if args.debug:
        if args.save_dir is not None:
            log_file = f"{args.s}-debug.log"
        else:
            log_file = "debug.log"
        handlers.append(logging.FileHandler(filename=log_file))
    logging.basicConfig(handlers=handlers)
    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.save_dir and not os.path.exists(os.path.join("runs", args.save_dir)):
        os.mkdir(os.path.join("runs", args.save_dir))
    main(
        n_runs=args.n,
        height=args.height,
        obj=args.obj,
        eplen=args.eplen,
        metric=args.metric,
        viewer=not (args.no_viewer),
        debug=args.debug,
        save_dir=args.save_dir,
        cem_iters=args.cem_iters,
        nerf_grasping=not args.no_nerf,
        use_grad_est=args.use_grad_est,
        robot_type=args.robot_type,
        use_true_normals=args.use_true_normals,
        grasp_data=args.grasp_data,
        grasp_idx=args.grasp_idx,
    )
