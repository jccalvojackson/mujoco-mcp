# server.py
import numpy as np
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage
from robot_descriptions import so_arm100_mj_description

from mujoco_mcp.robot import MujocoRobot

robot = MujocoRobot(so_arm100_mj_description.MJCF_PATH)

mcp = FastMCP("Robot State Control")


def _get_robot_description(robot: MujocoRobot) -> str:
    """Get description of robot joints and bounds for MCP tool."""
    bounds = robot.joint_bounds
    return (
        "Sets the state of the robot by providing a list of values for each joint. "
        f"The Robot has the following joints (with respective bounds): {bounds}. "
    )


def _side_by_side_images(images: list[np.ndarray]) -> Image:
    """Concatenate multiple images side by side."""
    concat_image: PILImage = PILImage.new(
        "RGB", (images[0].shape[1] * len(images), images[0].shape[0])
    )
    for i, image in enumerate(images):
        concat_image.paste(PILImage.fromarray(image), (i * image.shape[1], 0))
    return Image(data=concat_image.tobytes(), format="png")


def set_robot_state(state: list[float]) -> Image:
    """Set robot state and return rendered images."""
    robot.set_state(state)
    images = robot.render()
    robot.reset()  # to explicitly make server stateless
    return _side_by_side_images(images)


def register_robot_tools(mcp: FastMCP, robot: MujocoRobot) -> None:
    """Register robot control tools with the MCP server."""
    description = _get_robot_description(robot)
    mcp.tool(description=description)(set_robot_state)


register_robot_tools(mcp, robot)

# Main execution block - this is required to run the server
if __name__ == "__main__":
    mcp.run()
