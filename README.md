## MuJoCo MCP: Simulated Arm Pose Control

### What it is
An MCP server that lets an AI system control a MuJoCo arm to match a target pose, using images of a real arm as the reference.

### Quick start
- Install deps and create venv (uv):
  - `uv sync`
  - `source .venv/bin/activate`
- Run the server:
  - `python -m mujoco_mcp.server`

### MCP interface
- **Tool**: `set_robot_state(state: list[float])`
  - Input: joint positions (length = number of joints; bounds provided in the tool description)
  - Output: a WebP grid of simulated camera views
  - The server resets to a home pose after each call (stateless per call)
- **Prompt**: "Achieve pose"
  - Use your MCP-enabled agent to iteratively call the tool until the simulated views match the real images.

### Notes
- Default robot: `so_arm100` from `robot_descriptions`.
- Tuning cameras or render size: see `src/mujoco_mcp/robot.py`.

