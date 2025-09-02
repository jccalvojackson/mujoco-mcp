# MuJoCo MCP: AI-Controlled Robot Pose Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Pre-commit](https://github.com/jccalvojackson/mujoco-mcp/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/jccalvojackson/mujoco-mcp/actions/workflows/pre-commit.yml)
[![Tests](https://github.com/jccalvojackson/mujoco-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/jccalvojackson/mujoco-mcp/actions/workflows/tests.yml)

This [MCP](https://modelcontextprotocol.io/docs/getting-started/intro) (Model Context Protocol) server enables AI systems to control the joint positions of a simulated robot arm in [MuJoCo](https://github.com/google-deepmind/mujoco).

It is meant to be used by an agent to find a joint configuration that approximates the target configuration in the user provided reference images.

[It supports 50 different robot models](#available-robot-models).

### The prompt

It includes a robot kinematics informed prompt for the task of matching the joint configuration of reference images from a real robot. It includes details like the kinematic chain, joint limits, joint axis, etc. It aims to provide the agent with enough context to help it anticipate the effects of the different joint movements.

<details>
<summary><strong>Example prompt for SO100</strong></summary>

You are a robot pose matching assistant. Your task is to iteratively adjust a simulated robot's joint angles to match a target pose shown in reference images.

## Robot Configuration:
**Joint Order**: ['Elbow', 'Jaw', 'Pitch', 'Rotation', 'Wrist_Pitch', 'Wrist_Roll']

**Kinematic Chain**:
```
  Rotation_Pitch: Rotation
    Upper_Arm: Pitch
      Lower_Arm: Elbow
        Wrist_Pitch_Roll: Wrist_Pitch
          Fixed_Jaw: Wrist_Roll
            Moving_Jaw: Jaw
```

**Key Joints for Pose Matching**:
- **Wrist_Roll**: Largest range joint (range: -2.79 to 2.79 rad, ~5.6 rad range), axis: Y (pitch-like) - likely primary structural + likely end-effector control, fine orientation control, pitch motion - affects up/down orientation
- **Rotation**: Large range joint (range: -1.92 to 1.92 rad, ~3.8 rad range), axis: Y (pitch-like) - likely major orientation + pitch motion - affects up/down orientation
- **Pitch**: Very large range (range: -3.32 to 0.17 rad, ~3.5 rad range), axis: X (roll-like) - major structural impact + roll motion - affects twist/rotation
- **Wrist_Pitch**: Very large range (range: -1.66 to 1.66 rad, ~3.3 rad range), axis: X (roll-like) - major structural impact + likely end-effector control, roll motion - affects twist/rotation
- **Elbow**: Very large range (range: -0.17 to 3.14 rad, ~3.3 rad range), axis: X (roll-like) - major structural impact + roll motion - affects twist/rotation
- **Jaw**: (range: -0.17 to 1.75 rad, ~1.9 rad range), axis: Z (rotation/yaw-like) - yaw/rotation - affects left/right orientation

## Your Tools:
- `set_robot_state_and_render(state)`: Sets joint positions [Elbow, Jaw, Pitch, Rotation, Wrist_Pitch, Wrist_Roll] and returns a 4-view grid image

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
2. Set initial estimate: `set_robot_state_and_render([values_for_Elbow_Jaw_Pitch...])`
3. Compare results, identify differences
4. Adjust specific joints: Focus on the joints with the largest impact first
5. Repeat until satisfied with match

Begin by analyzing the reference image(s) and making your first pose estimate.
</details>

### The tool

Its only tool is the `set_robot_state_and_render` function which takes as input the joint positions of the robot and returns an image of the simulated robot at that configuration from several views.

```python
def set_robot_state_and_render(state: list[float]) -> Image:
```

See example below for an example output.

## Example of the MCP in action

After prompting cursor's claude 4 agent to match these photos:

<div style="display: flex; gap: 10px; align-items: center; justify-content: center; margin: 20px 0;">
  <img src="assets/test_15efc020ad_0.png" alt="Reference robot pose - View 1" style="max-width: 45%; height: auto;" />
  <img src="assets/test_b59c1bd94b_0.png" alt="Reference robot pose - View 2" style="max-width: 45%; height: auto;" />
</div>

*Reference images showing the target robot joint configuration from two different camera angles that the AI agent needs to match.*

It first tried this joint configuration:

![Initial joint configuration](assets/first_joint_state.png)

*Initial joint configuration attempt by the agent.*

On the fifth try, it stopped claiming success with a confidence that only an LLM can muster. While its final joint configuration is quite close to the reference images, is not quite there.

>Perfect! I have successfully achieved a very close match to the target pose shown in the reference images...

![Final joint configuration](assets/final_joint_state.png)

*Final 'successful' joint configuration by the agent.*

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jccalvojackson/mujoco-mcp.git
   cd mujoco-mcp
   ```

You can install it in your favourite MCP client (cursor, etc.) with the follwing config:

```json
    "mujoco-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/mujoco-mcp/",
        "/path/to/mujoco-mcp/src/mujoco_mcp/server.py"
      ]
    },
```

Alternatively, you can test it with the MCP inspector:

```bash
uv run mcp dev src/mujoco_mcp/server.py
```

## 🤖 Supported Robot Models

The MuJoCo MCP server supports **50 different robot models** that are compatible with MuJoCo's MJCF format. The default robot is `so_arm100_mj_description` (a 6-DOF robotic arm).

### Configuration

You can change the robot model by setting the environment variable:

```bash
export MUJOCO_MCP_ROBOT_NAME=<robot_name>
```

Or create a `.env` file in the project root:

```env
MUJOCO_MCP_ROBOT_NAME=<robot_name>
```

### Available Robot Models

<details>
<summary><strong>Click to see all 50 supported robot models</strong></summary>

| # | Robot Name | Type |
|---|------------|------|
| 1 | `a1_mj_description` | Quadruped |
| 2 | `ability_hand_mj_description` | Hand |
| 3 | `adam_lite_mj_description` | Humanoid |
| 4 | `aliengo_mj_description` | Quadruped |
| 5 | `allegro_hand_mj_description` | Hand |
| 6 | `aloha_mj_description` | Bimanual |
| 7 | `anymal_b_mj_description` | Quadruped |
| 8 | `anymal_c_mj_description` | Quadruped |
| 9 | `apollo_mj_description` | Humanoid |
| 10 | `arx_l5_mj_description` | Arm |
| 11 | `booster_t1_mj_description` | Humanoid |
| 12 | `cassie_mj_description` | Biped |
| 13 | `cf2_mj_description` | Drone |
| 14 | `dynamixel_2r_mj_description` | Arm |
| 15 | `elf2_mj_description` | Humanoid |
| 16 | `fr3_mj_description` | Arm |
| 17 | `g1_mj_description` | Humanoid |
| 18 | `gen3_mj_description` | Arm |
| 19 | `go1_mj_description` | Quadruped |
| 20 | `go2_mj_description` | Quadruped |
| 21 | `h1_mj_description` | Humanoid |
| 22 | `iiwa14_mj_description` | Arm |
| 23 | `jvrc_mj_description` | Humanoid |
| 24 | `leap_hand_mj_description` | Hand |
| 25 | `low_cost_robot_arm_mj_description` | Arm |
| 26 | `mujoco_humanoid_mj_description` | Humanoid |
| 27 | `n1_mj_description` | Humanoid |
| 28 | `op3_mj_description` | Humanoid |
| 29 | `panda_mj_description` | Arm |
| 30 | `piper_mj_description` | Arm |
| 31 | `robotiq_2f85_mj_description` | Gripper |
| 32 | `robotiq_2f85_v4_mj_description` | Gripper |
| 33 | `rsk_mj_description` | Arm |
| 34 | `sawyer_mj_description` | Arm |
| 35 | `shadow_dexee_mj_description` | Hand |
| 36 | `shadow_hand_mj_description` | Hand |
| 37 | `skydio_x2_mj_description` | Drone |
| 38 | `so_arm100_mj_description` | Arm (Default) |
| 39 | `so_arm101_mj_description` | Arm |
| 40 | `spot_mj_description` | Quadruped |
| 41 | `stretch_3_mj_description` | Mobile manipulator |
| 42 | `stretch_mj_description` | Mobile manipulator |
| 43 | `talos_mj_description` | Humanoid |
| 44 | `ur10e_mj_description` | Arm |
| 45 | `ur5e_mj_description` | Arm |
| 46 | `viper_mj_description` | Arm |
| 47 | `widow_mj_description` | Arm |
| 48 | `xarm7_mj_description` | Arm |
| 49 | `yam_mj_description` | Humanoid |
| 50 | `z1_mj_description` | Arm |

</details>

> **Note:** Each robot model has different joint configurations and ranges. The MCP tool will automatically adapt to the selected robot's joint structure and provide appropriate bounds information to AI agents.

## 🛠️ TODO

- [ ] Add standalone demo
- [ ] Measure pose matching performance and compare between at least two different agents and two different robot models


## 🤝 Contributing

Contributions are welcome and appreciated! Whether you're fixing bugs, adding features, or improving documentation, your help makes this project better.

### How to Contribute

1. **Fork the repository** and create your feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following the existing code style
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

### Ways to Contribute

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/jccalvojackson/mujoco-mcp/issues)
- 💡 **Suggest features** or improvements
- 🔧 **Fix issues** from the roadmap above
- 📚 **Improve documentation** and examples
- ⭐ **Star the repository** if you find it useful
- 🗣️ **Share the project** with others who might benefit

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/mujoco-mcp.git
cd mujoco-mcp

# Set up development environment
uv sync
source .venv/bin/activate

# Run tests (when available)
python -m pytest

# Run the server inspector for testing
uv run mcp dev src/mujoco_mcp/server.py
```

All contributions, no matter how small, are valued and appreciated!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
