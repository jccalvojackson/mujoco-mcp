import mujoco
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

from mujoco_mcp.config import config
from mujoco_mcp.image_utils import get_images_as_grid
from mujoco_mcp.robot import MujocoRobot

robot = MujocoRobot(config)


mujoco_mcp_server = FastMCP("Robot State Control")


def _get_kinematic_chain_info(mjcf_path: str) -> str:
    """Extract kinematic chain information from the robot model."""
    try:
        spec = mujoco.MjSpec.from_file(mjcf_path)
        bodies = spec.worldbody.bodies

        # Build kinematic chain description
        chain_info = []

        def traverse_body(body, depth=0):
            indent = "  " * depth
            body_name = body.name or "unnamed"

            # Get joints in this body
            joints_in_body = [joint.name for joint in body.joints if joint.name]

            if joints_in_body:
                chain_info.append(f"{indent}{body_name}: {', '.join(joints_in_body)}")
            elif depth > 0:  # Don't include root body if no joints
                chain_info.append(f"{indent}{body_name}")

            # Recursively traverse children
            for child in body.bodies:
                traverse_body(child, depth + 1)

        # Start traversal from worldbody
        for body in bodies:
            traverse_body(body)

        return "\n".join(chain_info) if chain_info else "No kinematic chain found"
    except Exception as e:
        return f"Could not extract kinematic chain: {e}"


def _get_robot_description(robot: MujocoRobot) -> str:
    """Get description of robot joints and bounds for MCP tool."""
    bounds = robot.joint_bounds
    return (
        "Sets the state of the robot by providing a list of values for each joint. "
        f"The Robot has the following joints (with respective bounds): {bounds}. "
    )


def pil_to_mcp_image(
    image: PILImage,
) -> Image:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=80, optimize=True)
    return Image(data=buffer.getvalue(), format="webp")


@mujoco_mcp_server.tool(description=_get_robot_description(robot))
def set_robot_state_and_render(state: list[float]) -> Image:
    """Set robot state and return rendered images."""
    robot.set_state(state)
    images = robot.render()
    robot.reset()  # to explicitly make server stateless
    pil_image = get_images_as_grid(images)
    return pil_to_mcp_image(pil_image)


@mujoco_mcp_server.prompt(title="Achieve pose")
def achieve_pose() -> str:
    return get_achieve_pose_prompt(robot)


def get_achieve_pose_prompt(robot: MujocoRobot) -> str:
    # Get robot-specific information
    joint_bounds = robot.joint_bounds
    joint_names = robot.joint_names
    kinematic_chain = _get_kinematic_chain_info(robot.config.robot_mjcf_path)

    return f"""You are a robot pose matching assistant. Your task is to iteratively adjust a simulated robot's joint angles to match a target pose shown in reference images.

## Robot Configuration:
**Joint Order**: {joint_names}
**Joint Bounds**: {joint_bounds}

**Kinematic Chain**:
```
{kinematic_chain}
```

## Your Tools:
- `set_robot_state_and_render(state)`: Sets joint positions [{", ".join(joint_names)}] and returns a 4-view grid image

## Strategy:
1. **Analyze**: Compare reference image(s) with current simulated robot pose
2. **Explore systematically**: Try different joint combinations - don't assume any joint should be zero
3. **Iterate**: Make adjustments based on visual comparison
4. **Refine**: Continue until poses match across all camera angles

## Critical Principles:

**Kinematic Chain Impact**:
- **Base joints** (early in chain): Small changes affect entire arm - massive leverage
- **Middle joints**: Create major structural changes in arm configuration  
- **End-effector joints** (late in chain): Fine-tune final positioning but still crucial

**Joint Exploration**:
- **Try both positive AND negative values** - don't assume joints should be neutral
- **Use full joint ranges** - target poses may require extreme values near limits
- **Every joint matters** - even small base rotations can be critical
- **Joint coupling** - changing one joint affects all subsequent joints in the chain

**Visual Comparison Strategy**:
- Compare overall arm shape and structure first
- Check end-effector position and orientation
- Verify pose matches from multiple camera angles
- Look for subtle differences in joint angles

## Systematic Approach:
1. **Start broad**: Make initial estimate covering overall pose structure
2. **Refine iteratively**: Adjust joints that show biggest visual mismatch
3. **Don't ignore any joint**: Even if a joint seems unimportant, try non-zero values
4. **Use incremental steps**: Typically 0.1-0.5 radian adjustments for fine-tuning

## Example Workflow:
```
1. Observe reference pose
2. Set initial estimate: set_robot_state_and_render([joint1, joint2, ...])
3. Compare result with reference
4. Identify biggest differences
5. Adjust relevant joints and repeat
```

Begin by analyzing the reference image(s) and making your first systematic pose estimate."""


# Main execution block - this is required to run the server
if __name__ == "__main__":
    mujoco_mcp_server.run(transport="stdio")
