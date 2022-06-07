import numpy as np
import pinocchio as pin

TIP_NAMES = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]


def create_trifinger(urdf_filename):
    """Builds a pinocchio model of the trifinger robot, and sets up its data."""
    model = pin.buildModelFromUrdf(urdf_filename)
    data = model.createData()

    return model, data


def get_tip_ids(model):
    """Helper function to get the pinocchio frame ids for the fingertips."""
    return [model.getFrameId(frame_name) for frame_name in TIP_NAMES]


def get_fingertip_pos(model, data, q):
    """Computes forward kinematics for fingertip positions."""
    tip_ids = get_tip_ids(model)

    # First run pinocchio forward kinematics to update joint data.
    pin.framesForwardKinematics(model, data, q)

    # Return fingertip positions.
    return np.concatenate(
        [pin.updateFramePlacement(model, data, tt).translation for tt in tip_ids],
        axis=0,
    )


def get_fingertip_pos_vel(model, data, q, dq):
    """Computes first-order forward kinematics to compute fingertip positions + velocities."""
    tip_ids = get_tip_ids(model)

    # First run pinocchio forward kinematics to update joint data.
    pin.forwardKinematics(model, data, q, dq)

    # Compute fingertip placements.
    p = np.concatenate(
        [pin.updateFramePlacement(model, data, tt).translation for tt in tip_ids],
        axis=0,
    )

    # Compute fingertip velocities (expressed in world frame!).
    v = np.concatenate(
        [
            pin.getFrameVelocity(model, data, tt, pin.ReferenceFrame.WORLD).linear
            for tt in tip_ids
        ],
        axis=0,
    )

    return p, v


def get_fingertip_jacobian(model, data, q):
    """Returns Jacobian (via pinocchio) for trifinger in configuration q."""
    tip_ids = get_tip_ids(model)
    pin.framesForwardKinematics(model, data, q)

    # Compute Jacobian of every fingertip frame, expressed in world frame.
    tip_jacs = [
        pin.computeFrameJacobian(model, data, q, tt, pin.ReferenceFrame.WORLD)
        for tt in tip_ids
    ]

    # Stack fingertip Jacobians to form a 9x9 matrix.
    return np.concatenate([tj[:3, :] for tj in tip_jacs], axis=0)


def add_collision_frames(model):
    """TODO: add operational frames to robot model to be used for collision terms."""

    raise NotImplementedError
