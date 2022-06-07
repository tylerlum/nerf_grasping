import logging
import os
import pdb
import time

import sim_trifinger as sim
from nerf_grasping.viz_utils import img_dir_to_vid


def run_robot_control(tf, height, eplen=250, debug=False, save_dir=None):
    count = 0  # skip off step
    save_freq = 1
    while not tf.gym.query_viewer_has_closed(tf.viewer) and count < eplen + 30:
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
        # obj_quat = tf.object.rb_states[0, 3:7]
        # logging.info(f"Object quaternion: {obj_quat}")
        # if height < obj_height :  # z-dim of object pose
        #     print(f"final height: {obj_height.numpy().item()}")
        #     return True
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
    robot=True,
    use_true_normals=False,
):
    if obj == "banana":
        obj = sim.Banana
    elif obj == "box":
        obj = sim.Box
    elif obj == "teddy_bear":
        obj = sim.TeddyBear
        obj.use_centroid = True
    elif obj == "powerdrill":
        obj = sim.PowerDrill
    success = False
    succ_total = 0
    tf = sim.TriFingerEnv(
        viewer=viewer,
        robot=robot,
        Obj=obj,
        use_nerf_grasping=nerf_grasping,
        use_residual_dirs=True,
        metric=metric,
        target_height=height,
        cem_iters=cem_iters,
        use_grad_est=use_grad_est,
        use_true_normals=use_true_normals,
    )
    for i in range(n_runs):
        if save_dir:
            save_path = f"runs/{save_dir}/{i}"
        else:
            save_path = None
        tf.reset()
        success = run_robot_control(
            tf=tf, height=height, eplen=eplen, debug=debug, save_dir=save_path
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
    parser.add_argument("--n", help="number of runs to evaluate", default=10, type=int)
    parser.add_argument("--h", help="goal height to lift to", default=0.07, type=float)
    parser.add_argument(
        "--m", "--metric", help="grasping metric to eval with", default="l1"
    )
    parser.add_argument("--o", "--obj", help="object to use", default="banana")
    parser.add_argument(
        "--eplen", help="length of episodes for each run", default=750, type=int
    )
    parser.add_argument("--d", "--debug", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--s", "--save_dir", default=None)
    parser.add_argument("--cem_iters", default=10)
    parser.add_argument("--no_nerf", action="store_true")
    parser.add_argument("--use_grad_est", action="store_true")
    parser.add_argument("--no_robot", action="store_true")
    parser.add_argument("--use_true_normals", action="store_true")

    args = parser.parse_args()
    handlers = [logging.StreamHandler()]
    if args.s:
        log_file = f"{args.s}-debug.log"
    else:
        log_file = "debug.log"
    handlers.append(logging.FileHandler(filename=log_file))
    logging.basicConfig(handlers=handlers)
    if args.d:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.s and not os.path.exists(os.path.join("runs", args.s)):
        os.mkdir(os.path.join("runs", args.s))
    main(
        n_runs=args.n,
        height=args.h,
        obj=args.o,
        eplen=args.eplen,
        metric=args.m,
        viewer=not (args.no_viewer),
        debug=args.d,
        save_dir=args.s,
        cem_iters=args.cem_iters,
        nerf_grasping=not args.no_nerf,
        use_grad_est=args.use_grad_est,
        robot=not args.no_robot,
        use_true_normals=args.use_true_normals,
    )
