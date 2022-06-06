import numpy as np
import pinocchio as pin

ROBOT_URDF = "assets/trifinger/robot_properties_fingers/urdf/trifinger_with_stage.urdf"
TIP_NAMES = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]


def create_trifinger():
    model = pin.buildModelFromUrdf(urdf_filename)
    data = model.createData()

    return model, data


def get_tip_ids(model):
    return [model.getFrameId(frame_name) for frame_name in TIP_NAMES]


def get_fingertip_pos(model, data, q):
    """Computes forward kinematics for fingertip positions."""
    tip_ids = get_tip_ids(model)
    pin.framesForwardKinematics(model, data, q)

    return np.concatenate(
        [pin.updateFramePlacement(model, data, tt).translation for tt in tip_ids],
        axis=0,
    )


def get_fingertip_pos_vel(model, data, q, dq):
    tip_ids = get_tip_ids(model)
    pin.framesForwardKinematics(model, data, q)

    p = np.concatenate(
        [pin.updateFramePlacement(model, data, tt).translation for tt in tip_ids],
        axis=0,
    )

    v = np.concatenate(
        [pin.getFrameVelocity(model, data, tt).linear for tt in tip_ids], axis=0
    )

    return p, v


def get_fingertip_jacobian(model, data, q):
    """Returns Jacobian (via pinocchio) for trifinger in configuration q."""
    tip_ids = get_tip_ids(model)
    pin.framesForwardKinematics(model, data, q)

    tip_jacs = [
        pin.computeFrameJacobian(
            model, data, q, tt, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        for tt in tip_ids
    ]

    return np.concatenate([tj[:3, :] for tj in tip_jacs], axis=0)


def add_collision_frames(model):

    raise NotImplementedError
