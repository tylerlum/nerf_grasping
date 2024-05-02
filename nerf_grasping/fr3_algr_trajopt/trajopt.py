import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydrake.geometry import (
    CollisionFilterDeclaration,
    GeometrySet,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import (
    DistanceConstraint,
    MinimumDistanceLowerBoundConstraint,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import (
    SnoptSolver,
    SolverId,
    SolverOptions,
)
from pydrake.systems.framework import DiagramBuilder

import nerf_grasping
from typing import Optional

# ############################# #
# COLLISION FILTERING FUNCTIONS #
# ############################# #

# see: https://stackoverflow.com/questions/76783635/how-to-allow-collisions-based-on-srdf-file


def get_collision_geometries(plant, body_name):
    try:
        return plant.GetCollisionGeometriesForBody(plant.GetBodyByName(body_name))
    except RuntimeError as e:
        print(f"Could not find {body_name}")
        return


def disable_collision(plant, collision_filter_manager, allowed_collision_pair):
    declaration = CollisionFilterDeclaration()
    set1 = GeometrySet()
    set2 = GeometrySet()
    set1_geometries = get_collision_geometries(plant, allowed_collision_pair[0])
    if set1_geometries is None:
        return
    set2_geometries = get_collision_geometries(plant, allowed_collision_pair[1])
    if set2_geometries is None:
        return
    set1.Add(set1_geometries)
    set2.Add(set2_geometries)
    declaration.ExcludeBetween(set1, set2)
    collision_filter_manager.Apply(declaration)


def load_srdf_disabled_collisions(srdf_file, plant, collision_filter_manager):
    tree = ET.parse(srdf_file)
    robot = tree.getroot()
    for disable_collisions in robot.iter("disable_collisions"):
        allowed_collision_pair = [
            disable_collisions.get("link1"),
            disable_collisions.get("link2"),
        ]
        disable_collision(plant, collision_filter_manager, allowed_collision_pair)


# ########################## #
# DRAKE DEBUGGING VISUALIZER #
# ########################## #


def PublishPositionTrajectory(
    trajectory, root_context, plant, visualizer, time_step=1.0 / 33.0
):
    """
    https://github.com/RussTedrake/manipulation/blob/8c3e596528c439214d63926ba011522fdf25c04a/manipulation/meshcat_utils.py#L454
    """
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)

    visualizer.StartRecording(False)

    for t in np.append(
        np.arange(trajectory.start_time(), trajectory.end_time(), time_step),
        trajectory.end_time(),
    ):
        root_context.SetTime(t)
        plant.SetPositions(plant_context, trajectory.value(t))
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()


@dataclass
class TrajOptParams:
    num_control_points: int = 25
    # min_duration: float = 5.0
    # max_duration: float = 10.0
    duration_cost: float = 1e0
    path_length_cost: float = 1e1
    # keep_hand_open: bool = False
    presolve_no_collision: bool = True
    min_self_coll_dist: float = 1e-4
    influence_dist: float = 1e-3
    use_lqr_cost: bool = True
    lqr_pos_weight: float = 1e-2
    lqr_vel_weight: float = 1e0
    nerf_frame_offset: float = 0.6
    s_start_self_col: float = 0.5


ALLEGRO_ROS2_PATH = (
    Path(nerf_grasping.get_package_root()) / "fr3_algr_ik" / "allegro_ros2"
)


class AllegroFR3TrajOpt:
    """
    Container for the Allegro FR3 trajectory optimization problem.
    """

    def __init__(
        self,
        q0: np.ndarray,
        qf: np.ndarray,
        cfg: TrajOptParams = TrajOptParams(),
        mesh_path: Optional[Path] = None,
        visualize: bool = False,
        verbose: bool = False,
    ) -> None:
        self.cfg = cfg
        self.mesh_path = mesh_path
        self.visualize = visualize
        self.verbose = verbose
        self.mesh_name = self.mesh_path.stem if self.mesh_path is not None else None

        if mesh_path is not None and not mesh_path.exists():
            print(f"WARNING: Mesh path {mesh_path} does not exist")

        self.setup_vanilla_allegro_fr3()
        self.setup_traj_opt(q0, qf)

    def setup_vanilla_allegro_fr3(self):
        # boilerplate for setting up arm and hand
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=0.001
        )
        self.parser = Parser(self.plant, self.scene_graph)
        self.parser.package_map().Add("allegro_ros2", str(ALLEGRO_ROS2_PATH))
        self.parser.AddModels(str(ALLEGRO_ROS2_PATH / "models/fr3_algr_zed2i.sdf"))

        # set up object
        # [TODO] do this properly?

        # add object to the plant
        if self.mesh_path is not None and self.mesh_name is not None:
            object_handle = self.parser.AddModels(str(self.mesh_path))[0]
            obj_body = self.plant.GetBodyByName(self.mesh_name)
            X_table_obj = RigidTransform(
                RotationMatrix(),
                np.array([self.cfg.nerf_frame_offset, 0.0, 0.0]),
            )
            X_table_W = RigidTransform(
                np.array(
                    [
                        [0.99992233, -0.00352763, 0.01195353, -0.00418759],
                        [0.00357804, 0.99998479, -0.00419804, 0.00665965],
                        [-0.01193854, 0.00424049, 0.99991974, -0.00257641],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )  # [TODO] Feb. 20, 2024 - manually computed fix in error between the robot and the table
            X_manual_compensation = RigidTransform(
                RotationMatrix(),
                np.array([-0.01, 0.01, 0.0]),
            )  # shift the object 1cm +x, +y manually
            X_WO = X_table_W.inverse() @ X_table_obj @ X_manual_compensation
            self.plant.WeldFrames(self.plant.world_frame(), obj_body.body_frame(), X_WO)

        self.plant.Finalize()

        if self.visualize:
            self.setup_visualization()

        self.diagram = self.builder.Build()
        self.diag_context = self.diagram.CreateDefaultContext()
        self.sg_context = self.scene_graph.GetMyMutableContextFromRoot(
            self.diag_context
        )
        self.plant_context = self.plant.GetMyContextFromRoot(self.diag_context)

        # collision filtering select pairs from srdf
        self.cfm = self.scene_graph.collision_filter_manager(self.sg_context)
        load_srdf_disabled_collisions(
            str(ALLEGRO_ROS2_PATH / "models/fr3_algr_zed2i.srdf"), self.plant, self.cfm
        )

        # collision filtering fingertips and object
        if self.mesh_name is not None:
            for geom_name in [
                "algr_rh_if_ds",
                "algr_rh_mf_ds",
                "algr_rh_rf_ds",
                "algr_rh_th_ds",
            ]:
                allowed_collision_pair = [self.mesh_name, geom_name]
                disable_collision(self.plant, self.cfm, allowed_collision_pair)

        self.qo_port = self.scene_graph.get_query_output_port()
        self.query_object = self.qo_port.Eval(self.sg_context)
        self.inspector = self.query_object.inspector()
        self.col_cands = list(self.inspector.GetCollisionCandidates())

    def setup_visualization(self):
        self.meshcat = StartMeshcat()
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder,
            self.scene_graph,
            self.meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration),
        )
        self.collision_visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder,
            self.scene_graph,
            self.meshcat,
            MeshcatVisualizerParams(
                prefix="collision", role=Role.kProximity, visible_by_default=False
            ),
        )

    def setup_traj_opt(self, q0: np.ndarray, qf: np.ndarray):
        # setting up the program
        self.q0 = q0
        self.qf = qf
        self.plant.SetPositions(self.plant_context, q0)
        self.num_q = self.plant.num_positions()
        self.trajopt = KinematicTrajectoryOptimization(
            self.plant.num_positions(), self.cfg.num_control_points, duration=5.0
        )
        self.prog = self.trajopt.get_mutable_prog()

        # Add velocity bounds
        self.trajopt.AddVelocityBounds(
            0.4 * self.plant.GetVelocityLowerLimits(),  # [DEBUG] slow down motion
            0.4 * self.plant.GetVelocityUpperLimits(),
        )

        # Add joint limits
        # for the hand limits, make sure q0 and qf are included in them
        lb = self.plant.GetPositionLowerLimits()
        ub = self.plant.GetPositionUpperLimits()
        lb[7:] = np.minimum(lb[7:], q0[7:])
        ub[7:] = np.maximum(ub[7:], q0[7:])
        lb[7:] = np.minimum(lb[7:], qf[7:])
        ub[7:] = np.maximum(ub[7:], qf[7:])
        self.trajopt.AddPositionBounds(lb=lb, ub=ub)

        # Add duration / path length costs.
        self.trajopt.AddDurationCost(self.cfg.duration_cost)

        if not self.cfg.use_lqr_cost:
            self.trajopt.AddPathLengthCost(self.cfg.path_length_cost)
        else:
            for tt in range(self.cfg.num_control_points - 1):
                # [TODO] penalize deviation from linearly interpolated traj instead of qf?
                pos_err = self.trajopt.control_points()[:, tt] - qf
                pos_cost = np.dot(pos_err, self.cfg.lqr_pos_weight * pos_err)
                self.prog.AddQuadraticCost(pos_cost)
                velocities = (
                    self.trajopt.control_points()[:, tt + 1]
                    - self.trajopt.control_points()[:, tt]
                ) / self.cfg.num_control_points  # Not scaled by time.

                vel_cost = np.dot(velocities, self.cfg.lqr_vel_weight * velocities)
                self.prog.AddQuadraticCost(vel_cost)

        # Start and end from rest.
        self.trajopt.AddPathVelocityConstraint(
            np.zeros((self.num_q, 1)), np.zeros((self.num_q, 1)), 0
        )
        self.trajopt.AddPathVelocityConstraint(
            np.zeros((self.num_q, 1)), np.zeros((self.num_q, 1)), 1
        )

        # Add initial and final constraints for arm.
        for ii in range(7):
            self.prog.AddConstraint(
                self.trajopt.control_points()[ii, 0] == q0[ii]
            ).evaluator().set_description(f"initial arm joint: {ii}")
            self.prog.AddConstraint(
                self.trajopt.control_points()[ii, -1] == qf[ii]
            ).evaluator().set_description(f"final arm joint: {ii}")

        # Add constraints for hand.
        # if self.cfg.keep_hand_open:
        #     # Set hand to be at end pose for whole trajectory.
        #     for tt in range(self.cfg.num_control_points):
        #         for ii in range(7, self.num_q):
        #             self.prog.AddConstraint(
        #                 self.trajopt.control_points()[ii, tt] == qf[ii]
        #             ).evaluator().set_description(f"hand joint: {ii}, time: {tt}")
        # else:

        # constrain start / end of hand.
        for ii in range(7, self.num_q):
            self.prog.AddConstraint(
                self.trajopt.control_points()[ii, 0] == q0[ii]
            ).evaluator().set_description(f"initial hand joint: {ii}")
            self.prog.AddConstraint(
                self.trajopt.control_points()[ii, -1] == qf[ii]
            ).evaluator().set_description(f"final hand joint: {ii}")

        init_guess = np.linspace(q0, qf, self.cfg.num_control_points).T

        # Setup self collision constraints.
        self.self_collision_constraint = MinimumDistanceLowerBoundConstraint(
            self.plant,
            self.cfg.min_self_coll_dist,
            self.plant_context,
            influence_distance_offset=self.cfg.influence_dist,
        )

        # Make output verbose.
        self.solver = SnoptSolver()
        # self.prog.SetSolverOption(SolverType.kSnopt, "Print file", "snopt_output.txt")

    def impose_self_collision_constraints(self):
        # the hand may start in self-collision and need to be opened, so we ignore those
        # constraints early on
        evaluate_at_s = np.linspace(0, 1, self.cfg.num_control_points)
        for s in evaluate_at_s:
            if s >= self.cfg.s_start_self_col:
                self.trajopt.AddPathPositionConstraint(
                    self.self_collision_constraint, s
                )

    def impose_finger_obj_collision_constraints(self):
        if self.mesh_name is None:
            return

        # impose a constraint that the fingertips are not in
        # collision with the object
        evaluate_at_s = np.linspace(0, 1, self.cfg.num_control_points)
        for s in evaluate_at_s:
            finger_names = [
                "algr_rh_if_ds",
                "algr_rh_mf_ds",
                "algr_rh_rf_ds",
                "algr_rh_th_ds",
            ]
            obj_name = self.mesh_name
            obj_body = self.plant.GetBodyByName(obj_name)
            obj_gid = self.plant.GetCollisionGeometriesForBody(obj_body)[0]
            for fname in finger_names:
                finger_body = self.plant.GetBodyByName(fname)
                finger_gid = self.plant.GetCollisionGeometriesForBody(finger_body)[0]
                pair = (obj_gid, finger_gid)
                constraint = DistanceConstraint(
                    self.plant,
                    pair,
                    self.plant_context,
                    0.01,
                    np.inf,
                )
                self.trajopt.AddPathPositionConstraint(constraint, s)

    def solve(self):
        options = SolverOptions()
        options.SetOption(SolverId("snopt"), "time limit", 10.0)
        options.SetOption(SolverId("snopt"), "timing level", 3)
        self.impose_finger_obj_collision_constraints()

        if self.cfg.presolve_no_collision:
            start = time.time()
            self.plant.SetPositions(self.plant_context, self.q0)

            # set initial guess to just be at final position
            # _knots = np.linspace(0, 5.0, self.cfg.num_control_points - 2)
            # knots = np.concatenate([[0.0, 0.0, 0.0], _knots, [5.0, 5.0, 5.0]])
            # basis = BsplineBasis(4, knots)
            # # control_points = np.stack([self.qf[..., None]] * self.cfg.num_control_points)
            # control_points = np.linspace(self.q0, self.qf, self.cfg.num_control_points)[..., None]
            # q_traj = BsplineTrajectory(basis, control_points)
            # self.trajopt.SetInitialGuess(q_traj)
            self.result = self.solver.Solve(self.prog, solver_options=options)
            end = time.time()
            print(f"Presolve took {end - start} seconds")
            self.trajopt.SetInitialGuess(
                self.trajopt.ReconstructTrajectory(self.result)
            )
            if not self.result.is_success():
                print("Presolve failed")
                # get_logger("presolve").info(self.result.GetInfeasibleConstraintNames(self.prog))
                details = self.result.get_solver_details()
                print(f"Details info: {details.info}")
                # self.introspect_collision_failure()
            else:
                print("Presolve succeeded")

        self.impose_self_collision_constraints()

        start = time.time()
        self.result = self.solver.Solve(self.prog, solver_options=options)
        end = time.time()

        print(f"Trajopt took {end - start} seconds")
        if not self.result.is_success():
            print("Trajectory optimization failed")
            # print(self.result.get_solver_id().name())
            if self.verbose:
                self.introspect_collision_failure()
        else:
            print("Trajectory optimization succeeded")

        if self.visualize:
            PublishPositionTrajectory(
                self.trajopt.ReconstructTrajectory(self.result),
                self.diag_context,
                self.plant,
                self.visualizer,
            )
            self.collision_visualizer.ForcedPublish(
                self.collision_visualizer.GetMyContextFromRoot(self.diag_context)
            )

    def introspect_collision_failure(self):
        spline = self.trajopt.ReconstructTrajectory(self.result)
        qo_port = self.scene_graph.get_query_output_port()

        if self.visualize:
            PublishPositionTrajectory(
                spline, self.diag_context, self.plant, self.visualizer
            )
        control_points = np.array(spline.control_points()).squeeze(-1)

        for i in range(self.cfg.num_control_points):
            if i / self.cfg.num_control_points < self.cfg.s_start_self_col:
                continue
            print(f"ctrl point {i}")
            qi = control_points[i, :]
            self.plant.SetPositions(self.plant_context, qi)
            query_object = self.qo_port.Eval(
                self.sg_context
            )  # remake query obj each update
            inspector = query_object.inspector()
            col_cands = list(inspector.GetCollisionCandidates())
            for c in col_cands:
                geometry_id1 = c[0]
                geometry_id2 = c[1]
                name1 = self.plant.GetBodyFromFrameId(
                    inspector.GetFrameId(geometry_id1)
                ).name()
                name2 = self.plant.GetBodyFromFrameId(
                    inspector.GetFrameId(geometry_id2)
                ).name()

                if ("algr" in name1 and "obj" in name2) or (
                    "obj" in name1 and "algr" in name2
                ):
                    dist_lower = 0.01
                elif ("fr3" in name1 and "obj" in name2) or (
                    "obj" in name1 and "fr3" in name2
                ):
                    dist_lower = 0.01
                elif (
                    ("algr" in name1 and "table" in name2)
                    or ("table" in name1 and "algr" in name2)
                    or ("fr3" in name1 and "table" in name2)
                    or ("table" in name1 and "fr3" in name2)
                ):
                    dist_lower = 0.01
                else:
                    dist_lower = 0.01
                sdp = query_object.ComputeSignedDistancePairClosestPoints(
                    geometry_id1, geometry_id2
                )
                signed_distance = sdp.distance
                if signed_distance <= dist_lower:
                    print(name1)
                    print(name2)
                    print(f"{signed_distance}")

        print(f"{self.result.GetInfeasibleConstraintNames(self.prog)}")


DEFAULT_Q_FR3 = np.array(
    [
        1.76261055e-06,
        -1.29018439e00,
        0.00000000e00,
        -2.69272642e00,
        0.00000000e00,
        1.35254201e00,
        7.85400000e-01,
    ]
)
DEFAULT_Q_ALGR = np.array(
    [
        2.90945620e-01,
        7.37109400e-01,
        5.10859200e-01,
        1.22637060e-01,
        1.20125350e-01,
        5.84513500e-01,
        3.43829930e-01,
        6.05035000e-01,
        -2.68431900e-01,
        8.78457900e-01,
        8.49713500e-01,
        8.97218400e-01,
        1.33282830e00,
        3.47787830e-01,
        2.09215670e-01,
        -6.50969000e-03,
    ]
)


def solve_trajopt(
    q_fr3_0: np.ndarray,
    q_algr_0: np.ndarray,
    q_fr3_f: np.ndarray,
    q_algr_f: np.ndarray,
    cfg: TrajOptParams,
    mesh_path: Optional[Path] = None,
    visualize: bool = False,
    verbose: bool = False,
):
    """Trajectory optimization callback upon receiving candidate grasps."""

    # grasp sampling loop
    print("Generating pregrasp trajectory...")

    q_robot_f = np.concatenate((q_fr3_f, q_algr_f))
    q_robot_0 = np.concatenate((q_fr3_0, q_algr_0))  # current configuration

    # setting up the trajopt and solving
    trajopt = AllegroFR3TrajOpt(
        q0=q_robot_0,
        qf=q_robot_f,
        cfg=cfg,
        mesh_path=mesh_path,
        visualize=visualize,
        verbose=verbose,
    )
    trajopt.solve()
    opt_result = trajopt.result

    if not opt_result.is_success():
        raise RuntimeError("Trajectory optimization failed")

    spline = trajopt.trajopt.ReconstructTrajectory(opt_result)
    dspline = spline.MakeDerivative()
    T_traj = opt_result.GetSolution(
        trajopt.trajopt.duration()
    )  # rescale the traj length
    print("Trajectory successfully generated!")

    if visualize:
        print("=" * 80)
        print("Visualizing trajectory (need breakpoint to keep visualization alive)...")
        print("=" * 80)
        breakpoint()

    return spline, dspline, T_traj, trajopt


def main() -> None:
    from nerf_grasping.fr3_algr_ik.ik import solve_ik

    NERF_FRAME_OFFSET = 0.65
    cfg = TrajOptParams(
        num_control_points=21,
        min_self_coll_dist=0.005,
        influence_dist=0.01,
        nerf_frame_offset=NERF_FRAME_OFFSET,
        s_start_self_col=0.5,
        lqr_pos_weight=1e-1,
        lqr_vel_weight=20.0,
        presolve_no_collision=True,
    )
    X_W_H_0 = np.array(
        [
            [-0.40069854, 0.06362686, 0.91399777, 0.66515265],
            [-0.367964, 0.90242159, -0.22413731, 0.02321906],
            [-0.83907259, -0.4261297, -0.33818674, 0.29229766],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    q_algr_0 = np.array(
        [
            0.29094562,
            0.7371094,
            0.5108592,
            0.12263706,
            0.12012535,
            0.5845135,
            0.34382993,
            0.605035,
            -0.2684319,
            0.8784579,
            0.8497135,
            0.8972184,
            1.3328283,
            0.34778783,
            0.20921567,
            -0.00650969,
        ]
    )
    q_robot_0 = solve_ik(X_W_H_0, q_algr_0, visualize=False)

    X_W_H_f = X_W_H_0.copy()
    X_W_H_f[:3, 3] = np.array([0.56515265, 0.12321906, 0.19229766])
    q_algr_f = q_algr_0.copy()
    q_robot_f = solve_ik(X_W_H_f, q_algr_f, visualize=False)

    mesh_path = (
        Path(nerf_grasping.get_repo_root())
        / "experiments/2024-05-01_15-39-42/nerf_to_mesh/mug_330/coacd/decomposed.obj"
    )
    try:
        spline, dspline, T_traj, trajopt = solve_trajopt(
            q_fr3_0=q_robot_0[:7],
            q_algr_0=q_robot_0[7:],
            q_fr3_f=q_robot_f[:7],
            q_algr_f=q_robot_f[7:],
            cfg=cfg,
            mesh_path=mesh_path,
            visualize=True,
            verbose=False,
        )
        print("Trajectory optimization succeeded!")
    except RuntimeError as e:
        print("Trajectory optimization failed")


if __name__ == "__main__":
    main()
