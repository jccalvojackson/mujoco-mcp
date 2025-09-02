import mujoco
import numpy as np
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

from mujoco_mcp.config import config
from mujoco_mcp.robot import MujocoRobot

robot = MujocoRobot(config.robot_mjcf_path)
spec = mujoco.MjSpec.from_file(config.robot_mjcf_path)


mujoco_mcp_server = FastMCP("Robot State Control")


def _get_kinematic_chain_info() -> str:
    """Extract kinematic chain information from the robot model."""
    try:
        spec = mujoco.MjSpec.from_file(config.robot_mjcf_path)
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


def _get_enhanced_joint_analysis(robot: MujocoRobot) -> dict:
    """Extract detailed joint analysis for pose matching guidance."""
    try:
        joint_bounds = robot.joint_bounds
        joint_names = robot.joint_names

        # Analyze joint axes and types
        joint_analysis = {}

        for joint_name in joint_names:
            joint_lower, joint_upper = joint_bounds[joint_name]
            joint_range = joint_upper - joint_lower

            analysis = {
                "range": joint_range,
                "bounds": (joint_lower, joint_upper),
                "axis": None,
                "type": "unknown",
            }

            axis = robot._model.joint(joint_name).axis
            # Determine axis type
            if abs(axis[2]) > 0.9:  # Z-axis dominant
                analysis["axis"] = "Z (rotation/yaw-like)"
                analysis["type"] = "rotational"
            elif abs(axis[1]) > 0.9:  # Y-axis dominant
                analysis["axis"] = "Y (pitch-like)"
                analysis["type"] = "rotational"
            elif abs(axis[0]) > 0.9:  # X-axis dominant
                analysis["axis"] = "X (roll-like)"
                analysis["type"] = "rotational"
            else:
                analysis["axis"] = (
                    f"Custom ({axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f})"
                )
                analysis["type"] = "rotational"

            joint_analysis[joint_name] = analysis

        return joint_analysis
    except Exception:
        return {}


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
    # Get robot-specific information
    joint_bounds = robot.joint_bounds
    joint_names = robot.joint_names
    kinematic_chain = _get_kinematic_chain_info()
    joint_analysis = _get_enhanced_joint_analysis(robot)

    # Analyze joint characteristics for pose matching guidance
    key_joints = []
    joint_ranges = []
    for joint in joint_names:
        joint_lower, joint_upper = joint_bounds[joint]
        joint_range = joint_upper - joint_lower
        joint_ranges.append((joint, joint_range, joint_lower, joint_upper))

    # Sort by range to identify potentially important joints
    joint_ranges.sort(key=lambda x: x[1], reverse=True)

    # Analyze kinematic chain position to identify joint roles
    # Joints later in the chain (higher index) are more likely to control end-effector orientation
    num_joints = len(joint_names)

    for i, (joint, joint_range, joint_lower, joint_upper) in enumerate(joint_ranges):
        joint_description = f"(range: {joint_lower:.2f} to {joint_upper:.2f} rad, ~{joint_range:.1f} rad range)"

        # Find this joint's position in the kinematic chain
        joint_index = joint_names.index(joint)
        chain_position = (
            joint_index / (num_joints - 1) if num_joints > 1 else 0
        )  # 0 to 1

        # Add enhanced analysis information
        analysis_info = joint_analysis.get(joint, {})
        axis_info = analysis_info.get("axis", "unknown axis")

        # Add role-based annotations based on chain position, range, and axis
        role_hints = []
        if chain_position > 0.6:  # Later joints in chain
            role_hints.append("likely end-effector control")
        if (
            chain_position > 0.8 and joint_range > 1.0
        ):  # Very late joints with decent range
            role_hints.append("fine orientation control")

        # Add axis-specific hints
        if "pitch" in axis_info.lower():
            role_hints.append("pitch motion - affects up/down orientation")
        elif "roll" in axis_info.lower():
            role_hints.append("roll motion - affects twist/rotation")
        elif "yaw" in axis_info.lower() or "rotation" in axis_info.lower():
            role_hints.append("yaw/rotation - affects left/right orientation")

        # Build joint description with axis information
        full_description = f"{joint_description}, axis: {axis_info}"

        if i == 0:
            # Largest range joint - likely primary structural
            role = "likely primary structural"
            if role_hints:
                role += f" + {', '.join(role_hints)}"
            key_joints.append(
                f"- **{joint}**: Largest range joint {full_description} - {role}"
            )
        elif i == 1 and joint_range > 2.0:
            # Second largest with significant range
            role = "likely major orientation"
            if role_hints:
                role += f" + {', '.join(role_hints)}"
            key_joints.append(
                f"- **{joint}**: Large range joint {full_description} - {role}"
            )
        elif joint_range > 3.0:
            # Very large range
            role = "major structural impact"
            if role_hints:
                role += f" + {', '.join(role_hints)}"
            key_joints.append(
                f"- **{joint}**: Very large range {full_description} - {role}"
            )
        else:
            # Standard joint
            role = ""
            if role_hints:
                role = f" - {', '.join(role_hints)}"
            key_joints.append(f"- **{joint}**: {full_description}{role}")

    return f"""You are a robot pose matching assistant. Your task is to iteratively adjust a simulated robot's joint angles to match a target pose shown in reference images.

## Robot Configuration:
**Joint Order**: {joint_names}

**Kinematic Chain**:
```
{kinematic_chain}
```

**Key Joints for Pose Matching**:
{chr(10).join(key_joints)}

## Your Tools:
- `set_robot_state_and_render(state)`: Sets joint positions [{", ".join(joint_names)}] and returns a 4-view grid image

## Strategy:
1. **Analyze**: Compare the reference image(s) with the current simulated robot pose
2. **Plan**: Identify which joints need adjustment and estimate the direction/magnitude
3. **Iterate**: Make incremental adjustments, typically 0.1-0.3 radians per step
4. **Refine**: Continue until the poses closely match across all camera angles

## Systematic Joint Adjustment Approach:
1. **Start with structural joints** (largest range, early in chain): Set overall arm configuration
2. **Adjust intermediate joints**: Fine-tune arm positioning and reach
3. **End-effector orientation** (later joints in chain): Control final pose and orientation
4. **Iterative refinement**: Make small adjustments to joints that most affect the mismatch

## Key Guidelines:
- **For compact/folded poses**: Consider using extreme joint values near the limits for large-range joints
- **Start with structural joints**: Focus on joints with large ranges that create major pose changes first
- **Joint prioritization**: Large-range joints typically have the most impact on overall configuration
- **End-effector fine-tuning**: Use joints later in the kinematic chain for final orientation
- **Explore the space**: Don't assume moderate values - target poses may require extreme joint angles
- **Joint interaction awareness**: Changing one joint affects the position/orientation of all subsequent joints in the chain
- The robot resets to home position after each state setting, so provide complete joint states

## Critical Pose Analysis Points:
- **End-effector orientation**: Pay special attention to the final orientation of grippers/tools
- **Joint sign sensitivity**: Small sign changes in orientation joints can dramatically change end-effector direction
- **Axis-specific behavior**:
  - Pitch joints (Y-axis): Control up/down tilt - positive/negative can flip orientation dramatically
  - Roll joints (X-axis): Control twist/rotation around arm axis - small changes affect grip orientation
  - Yaw/Rotation joints (Z-axis): Control left/right swing - major structural changes
- **Range exploration**: Don't hesitate to try values near joint limits for dramatic pose changes
- **Incremental refinement**: After getting close, make small adjustments (±0.1-0.3 rad) to fine-tune
- **Kinematic coupling**: Later joints inherit the orientation of earlier joints - a pitch joint's effect depends on preceding rotations

## Pose Matching Tips:
- **Compact poses**: Try extreme values for large-range joints (near their limits)
- **Extended poses**: Use moderate values for structural joints
- **Different orientations**: Focus on joints later in the kinematic chain - they control end-effector orientation
- **Structural impact**: Focus on the largest-range joints first - they typically create the most dramatic pose changes
- **Orientation mismatch**: If structure looks right but orientation is wrong, adjust end-effector joints (later in chain)
- **Sign matters**: Try both positive and negative values for orientation joints - small changes can flip orientations

## Orientation Troubleshooting Strategy:
1. **Structure first, orientation second**: Get the overall arm configuration right before fine-tuning end-effector orientation
2. **Identify orientation joints**: Focus on joints later in the kinematic chain with pitch/roll/yaw axes
3. **Systematic sign testing**: If orientation is wrong, try flipping the sign of orientation joints (±0.5 to ±2.0 rad changes)
4. **Axis-aware adjustments**:
   - Wrong up/down angle → adjust pitch joints
   - Wrong twist/rotation → adjust roll joints
   - Wrong left/right direction → adjust yaw/rotation joints
5. **Range bracketing**: Try both extremes of an orientation joint's range to see the full effect

## Example workflow:
1. Observe reference pose and current simulated pose
2. Set initial estimate: `set_robot_state_and_render([values_for_{"_".join(joint_names[:3])}...])`
3. Compare results, identify differences
4. Adjust specific joints: Focus on the joints with the largest impact first
5. Repeat until satisfied with match

Begin by analyzing the reference image(s) and making your first pose estimate."""


# Main execution block - this is required to run the server
if __name__ == "__main__":
    mujoco_mcp_server.run(transport="stdio")
