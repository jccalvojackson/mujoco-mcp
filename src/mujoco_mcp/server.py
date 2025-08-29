import numpy as np
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

from mujoco_mcp.config import config
from mujoco_mcp.robot import MujocoRobot

robot = MujocoRobot(config.robot_mjcf_path)

mujoco_mcp_server = FastMCP("Robot State Control")


def _get_robot_description(robot: MujocoRobot) -> str:
    """Get description of robot joints and bounds for MCP tool."""
    bounds = robot.joint_bounds
    return (
        "Sets the state of the robot by providing a list of values for each joint. "
        f"The Robot has the following joints (with respective bounds): {bounds}. "
    )


def _get_images_as_grid(
    images: list[np.ndarray],
) -> Image:
    """Concatenate multiple images in a grid layout with WebP compression."""
    import math

    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images provided")

    MAX_IMAGES = 9
    if num_images > MAX_IMAGES:
        raise ValueError(f"Too many images: {num_images} > {MAX_IMAGES}")

    # Calculate grid dimensions for a more square layout
    if num_images <= 4:
        cols = min(2, num_images)
        rows = math.ceil(num_images / cols)
    else:
        # For 5+ images, prefer 3 columns for better aspect ratio
        cols = 3
        rows = math.ceil(num_images / cols)

    # Get dimensions of individual images
    img_height, img_width = images[0].shape[:2]

    # Create the grid canvas
    grid_width = cols * img_width
    grid_height = rows * img_height
    concat_image: PILImage = PILImage.new("RGB", (grid_width, grid_height))

    # Place images in grid
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        x_pos = col * img_width
        y_pos = row * img_height
        concat_image.paste(PILImage.fromarray(image), (x_pos, y_pos))

    from io import BytesIO

    buffer = BytesIO()
    concat_image.save(buffer, format="WEBP", quality=80, optimize=True)
    return Image(data=buffer.getvalue(), format="webp")


@mujoco_mcp_server.tool(description=_get_robot_description(robot))
def set_robot_state_and_render(state: list[float]) -> Image:
    """Set robot state and return rendered images."""
    robot.set_state(state)
    images = robot.render()
    robot.reset()  # to explicitly make server stateless
    return _get_images_as_grid(images)


@mujoco_mcp_server.prompt(title="Achieve pose")
def achieve_pose() -> str:
    return """You are a robot pose matching assistant. Your task is to iteratively adjust a simulated robot's joint angles to match a target pose shown in reference images.

## Your Tools:
- `set_robot_state_and_render(state)`: Sets joint positions and returns a 4-view grid image of the simulated robot

## Strategy:
1. **Analyze**: Compare the reference image(s) with the current simulated robot pose
2. **Plan**: Identify which joints need adjustment and estimate the direction/magnitude
3. **Iterate**: Make incremental adjustments, typically 0.1-0.3 radians per step
4. **Refine**: Continue until the poses closely match across all camera angles

## Key Guidelines:
- Start with large structural joints before fine-tuning end-effector orientation
- Use the n-camera grid to verify pose accuracy from multiple angles
- Make conservative adjustments to avoid overshooting
- Pay attention to both joint positions and overall arm configuration
- The robot resets to home position after each state setting, so provide complete joint states

## Example workflow:
1. Observe reference pose and current simulated pose
2. Set initial estimate: `set_robot_state_and_render([initial_values...])`
3. Compare results, identify differences
4. Adjust specific joints: `set_robot_state_and_render([adjusted_values...])`
5. Repeat until satisfied with match

Begin by analyzing the reference image(s) and making your first pose estimate."""


# Main execution block - this is required to run the server
if __name__ == "__main__":
    mujoco_mcp_server.run(transport="stdio")
