import re
import sys
from typing import Optional

import mujoco
import numpy as np
from loguru import logger
from pydantic import BaseModel

logger.remove()
logger.add(sys.stderr, level="DEBUG")


class RenderConfig(BaseModel):
    width: int = 480
    height: int = 320
    num_cameras: int = 4


def get_cameras_for_link(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    link_id: int,
    *,
    num_cameras: int,
) -> list[mujoco.MjvCamera]:
    # target point to frame: world-space position of the geom
    mujoco.mj_step(model, data)
    lookat = np.asarray(data.geom_xpos[link_id].copy(), dtype=float)

    # estimate a reasonable camera distance from geom size
    # Use L2 norm of geom size as a proxy for bounding radius
    geom_size = np.asarray(model.geom_size[link_id], dtype=float)
    bounding_radius = float(np.linalg.norm(geom_size))
    logger.debug(f"Bounding radius: {bounding_radius}")
    distance = max(0.6, 5.0 * bounding_radius)

    cameras: list[mujoco.MjvCamera] = []
    if num_cameras <= 0:
        return cameras

    # Simple single ring: evenly spaced azimuths at a fixed elevation above horizon
    fixed_elevation_deg = -30.0
    azimuths = np.linspace(0.0, 360.0, num_cameras, endpoint=False)

    for az in azimuths:
        cam = mujoco.MjvCamera()
        cam.lookat = lookat
        cam.distance = distance
        cam.azimuth = float(az)
        cam.elevation = fixed_elevation_deg
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
        xml_path: str,
        render_config: RenderConfig = RenderConfig(),
        home_position: np.ndarray | None = None,
        link_base: int = 0,
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._model.vis.global_.offwidth = render_config.width
        self._model.vis.global_.offheight = render_config.height
        self._data = mujoco.MjData(self._model)
        # TODO: try getting list of valid components names from mujoco or create enum
        self._actuators = _get_component_names(self._model, "actuator")
        logger.info(f"Actuators: {self._actuators}")
        self._actuator_ids = np.asarray(
            [self._model.actuator(actuator).id for actuator in self._actuators]
        )
        self._joints = _get_component_names(self._model, "joint")
        logger.info(f"Joints: {self._joints}")
        self._joint_ids = np.asarray(
            [self._model.joint(joint).id for joint in self._joints]
        )
        self._joint_ranges = np.stack(
            [self._model.joint(joint).range for joint in self._joints]
        )
        logger.info(f"Joint ranges: {self._joint_ranges}")
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_config = render_config
        self._cameras: list[mujoco.MjvCamera] = get_cameras_for_link(
            model=self._model,
            data=self._data,
            link_id=link_base,
            num_cameras=render_config.num_cameras,
        )
        self._home_position = home_position or np.zeros(len(self._joints))

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
    def joint_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            joint: (low.item(), high.item())
            for joint, (low, high) in zip(self._joints, self._joint_ranges)
        }

    def reset(self):
        self.set_state(self._home_position)

    def set_state(self, state: np.ndarray):
        """Set the robot state."""
        if len(state) != len(self._joints):
            raise ValueError(
                f"State must have {len(self._joints)} values, got {len(state)}"
            )
        state = np.clip(state, self._joint_ranges[:, 0], self._joint_ranges[:, 1])
        self._data.qpos[self._joint_ids] = state
        mujoco.mj_forward(self._model, self._data)

    def apply_action(self, action):
        if len(action) != len(self._actuator_ids):
            raise ValueError(
                f"Action must have {len(self._actuator_ids)} values, got {len(action)}"
            )
        """Apply the action to the robot."""
        # TODO: apply action during an amount of time on a loop
        action = np.clip(action, self._joint_ranges[:, 0], self._joint_ranges[:, 1])
        self._data.ctrl[self._actuator_ids] = action
        mujoco.mj_step(self._model, self._data)

    def get_robot_state(self) -> np.ndarray:
        """Get the current state of the robot."""
        joint_pos = self._data.qpos[self._joint_ids].astype(np.float32)
        return joint_pos


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    from robot_descriptions import so_arm100_mj_description

    robot = MujocoRobot(so_arm100_mj_description.MJCF_PATH)
    action = np.array([np.pi / 2, np.pi / 2, 0, 0, 0, 0])
    robot.set_state(action)
    logger.info(f"Robot state: {robot.get_robot_state()}")
    images = robot.render()
    for i, image in enumerate(images):
        image = Image.fromarray(image)
        image.save(f"image_{i}.png")
