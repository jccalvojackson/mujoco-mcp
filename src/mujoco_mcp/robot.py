import re
import sys
from typing import Annotated, Optional

import mujoco
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from mujoco_mcp.config import Config

logger.remove()
logger.add(sys.stderr, level="DEBUG")


class RenderConfig(BaseModel):
    width: int = 480
    height: int = 320
    num_cameras: Annotated[int, Field(ge=1, le=4)] = 4


def get_cameras(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    num_cameras: int = 4,
) -> list[mujoco.MjvCamera]:
    """
    Get cameras using photography-inspired angles.
    """
    MAXIMUM_NUMBER_OF_CAMERAS = 4
    if num_cameras > MAXIMUM_NUMBER_OF_CAMERAS:
        raise ValueError(
            f"Maximum number of cameras is {MAXIMUM_NUMBER_OF_CAMERAS}, got {num_cameras}"
        )
    # Get current arm configuration for adaptive positioning
    mujoco.mj_forward(model, data)

    # Find key points: base, elbow, end-effector
    base_pos: np.ndarray = data.xpos[0]  # Assuming base is body 0
    end_effector_pos: np.ndarray = data.xpos[-1]  # Last body is typically end-effector

    # Compute arm center and span for framing
    arm_center = (base_pos + end_effector_pos) / 2
    arm_span = np.linalg.norm(end_effector_pos - base_pos)
    distance = max(1.0, 2.0 * arm_span)  # Closer than current implementation

    # Photography-inspired camera angles
    camera_configs = [
        {"azimuth": 45, "elevation": -15, "name": "front-right"},  # Classic 3/4 view
        {"azimuth": 135, "elevation": -15, "name": "back-right"},  # Opposite 3/4 view
        {"azimuth": 0, "elevation": -25, "name": "front"},  # Straight front
        {"azimuth": 90, "elevation": -5, "name": "side"},  # Pure side view
    ]

    cameras = []
    for config in camera_configs[:num_cameras]:
        cam = mujoco.MjvCamera()
        cam.lookat = arm_center
        cam.distance = distance
        cam.azimuth = config["azimuth"]
        cam.elevation = config["elevation"]
        cameras.append(cam)

    return cameras


def _get_component_names(model: mujoco.MjModel, component_type: str) -> list[str]:
    try:
        getattr(model, component_type)()
    except KeyError as e:
        match = re.search(r"Valid names: (\[.*?\])", str(e))
        if match:
            try:
                import ast

                return ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                pass
        raise ValueError(f"Invalid component type: {component_type}")
    except Exception as e:
        raise ValueError(f"Error getting component names: {e}")


class MujocoRobot:
    def __init__(
        self,
        config: Config,
        render_config: RenderConfig = RenderConfig(),
        home_position: np.ndarray | None = None,
        link_base: int = 0,
    ):
        self._config = config
        self._model = mujoco.MjModel.from_xml_path(config.robot_mjcf_path)
        self._model.vis.global_.offwidth = render_config.width
        self._model.vis.global_.offheight = render_config.height
        self._data = mujoco.MjData(self._model)
        # TODO: try getting list of valid components names from mujoco or create enum
        self._actuators: list[str] = _get_component_names(self._model, "actuator")
        logger.info(f"Actuators: {self._actuators}")
        self._actuator_ids: np.ndarray = np.asarray(
            [self._model.actuator(actuator).id for actuator in self._actuators]
        )
        self._joints: list[str] = _get_component_names(self._model, "joint")
        logger.info(f"Joints: {self._joints}")
        self._joint_ids: np.ndarray = np.asarray(
            [self._model.joint(joint).id for joint in self._joints]
        )
        self._joint_ranges: np.ndarray = np.stack(
            [self._model.joint(joint).range for joint in self._joints]
        ).T
        logger.info(f"Joint bounds: {self.joint_bounds}")
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_config: RenderConfig = render_config
        self._cameras: list[mujoco.MjvCamera] = get_cameras(
            model=self._model,
            data=self._data,
            num_cameras=render_config.num_cameras,
        )
        self._home_position: np.ndarray = home_position or np.zeros(len(self._joints))

    def render(self) -> list[np.ndarray]:
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_config.height,
                width=self._render_config.width,
            )
        shots = []
        for camera in self._cameras:
            self._viewer.update_scene(self._data, camera=camera)
            shots.append(self._viewer.render())
        return shots

    def close(self) -> None:
        viewer = self._viewer
        if viewer is None:
            return

        viewer.close()

        self._viewer = None

    @property
    def joint_names(self) -> list[str]:
        return self._joints

    @property
    def config(self) -> Config:
        return self._config

    @classmethod
    def from_robot_name(cls, robot_name: str) -> "MujocoRobot":
        config = Config(robot_name=robot_name)
        return cls(config)

    @property
    def joint_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            joint: (low.item(), high.item())
            for joint, (low, high) in zip(self._joints, self._joint_ranges.T)
        }

    def reset(self):
        self.set_state(self._home_position)

    def set_state(self, state: np.ndarray):
        """Set the robot state."""
        if len(state) != len(self.joint_names):
            raise ValueError(
                f"State must have {len(self.joint_names)} values, got {len(state)}"
            )
        state = np.clip(state, *self._joint_ranges)
        self._data.qpos[self._joint_ids] = state
        mujoco.mj_forward(self._model, self._data)

    def apply_action(self, action):
        if len(action) != len(self._actuator_ids):
            raise ValueError(
                f"Action must have {len(self._actuator_ids)} values, got {len(action)}"
            )
        """Apply the action to the robot."""
        # TODO: apply action during an amount of time on a loop
        action = np.clip(action, *self._joint_ranges)
        self._data.ctrl[self._actuator_ids] = action
        mujoco.mj_step(self._model, self._data)

    def get_robot_state(self) -> np.ndarray:
        """Get the current state of the robot."""
        joint_pos = self._data.qpos[self._joint_ids].astype(np.float32)
        return joint_pos


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    robot = MujocoRobot.from_robot_name("so_arm100_mj_description")
    action = np.array([np.pi / 2, np.pi / 2, 0, 0, 0, 0])
    robot.set_state(action)
    body_name = "Fixed_Jaw"
    logger.info(f"Robot state: {robot.get_robot_state()}")
    images = robot.render()
    for i, image in enumerate(images):
        image = Image.fromarray(image)
        image.save(f"image_{i}.png")
