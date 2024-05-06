import numpy as np
from pydrake.all import (
    AddFrameTriadIllustration,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    JointSliders,
    Parser,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    RigidTransform,
    Role,
    RotationMatrix,
    StartMeshcat,
)
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
import pathlib


def solve_ik(
    X_W_H: np.ndarray,
    q_algr_pre: np.ndarray,
    visualize: bool = False,
    position_constraint_tolerance: float = 0.001,
    angular_constraint_tolerance: float = 0.05,
) -> np.ndarray:
    current_dir = pathlib.Path(__file__).parent.absolute()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("allegro_ros2", str(current_dir / "allegro_ros2"))
    parser.AddModels(str(current_dir / "allegro_ros2/models/fr3_algr_zed2i.urdf"))
    plant.Finalize()

    q_arm_home = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
    q_guess = np.concatenate((q_arm_home, q_algr_pre))
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, q_guess)
    ik_pre = InverseKinematics(plant, context)
    ik_pre.AddPositionConstraint(
        plant.GetFrameByName("algr_rh_palm"),
        np.zeros(3),
        plant.world_frame(),
        X_W_H[:3, -1] - position_constraint_tolerance,
        X_W_H[:3, -1] + position_constraint_tolerance,
    )  # location of the palm
    ik_pre.AddOrientationConstraint(
        plant.GetFrameByName("algr_rh_palm"),
        RotationMatrix(),  # the palm frame
        plant.world_frame(),
        RotationMatrix(X_W_H[:3, :3]),  # the desired palm orientation in the world
        angular_constraint_tolerance,  # angular constraint tolerance should be small
    )  # [TODO] this is causing issues!
    prog_pre = ik_pre.get_mutable_prog()
    prog_pre.AddBoundingBoxConstraint(
        q_guess[7:],
        q_guess[7:],
        ik_pre.q()[7:],
    )  # constrain the hand states
    prog_pre.SetInitialGuessForAllVariables(q_guess)
    result = Solve(prog_pre)
    if not result.is_success():
        raise RuntimeError("IK failed")
    q_star = result.GetSolution(ik_pre.q())
    if visualize:
        # starting meshcat + adding sliders
        meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration),
        )
        collision_visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(
                prefix="collision",
                role=Role.kProximity,
                visible_by_default=True,
            ),
        )
        sliders = builder.AddSystem(JointSliders(meshcat, plant, step=1e-4))
        diagram = builder.Build()

        sliders.SetPositions(q_star)
        sliders.Run(diagram, None)
    return q_star


def main() -> None:
    # CHANGE THESE
    ######################################
    X_W_H = np.array(
        [
            [-0.40069854, 0.06362686, 0.91399777, 0.66515265],
            [-0.367964, 0.90242159, -0.22413731, 0.02321906],
            [-0.83907259, -0.4261297, -0.33818674, 0.29229766],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    q_algr_pre = np.array(
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
    ######################################
    q_star = solve_ik(X_W_H, q_algr_pre, visualize=True)
    print(f"q_star: {q_star}")


if __name__ == "__main__":
    main()
