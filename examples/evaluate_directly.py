import tempfile
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import numpy as np
import typer
import weave
from agno.agent import Agent
from agno.media import Image, ImageArtifact
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.run.agent import RunOutput
from agno.tools import Toolkit
from agno.tools.function import ToolResult
from agno.utils.log import log_debug, logger
from PIL import Image as PILImage

import wandb
from mujoco_mcp.image_utils import get_images_as_grid
from mujoco_mcp.robot import MujocoRobot
from mujoco_mcp.server import get_achieve_pose_prompt, pil_to_mcp_image

WANDB_PROJECT_NAME = "evaluate_joint_configuration_agent"

weave.init(WANDB_PROJECT_NAME)


class ModelClass(StrEnum):
    CLAUDE = "claude"
    GEMINI = "gemini"


MODEL_NAMES = {
    ModelClass.CLAUDE: "claude-sonnet-4-20250514",
    ModelClass.GEMINI: "gemini-2.5-pro",
}


def get_model(model_class: ModelClass) -> Claude | Gemini:
    match model_class:
        case ModelClass.CLAUDE:
            return Claude(id=MODEL_NAMES[model_class])
        case ModelClass.GEMINI:
            return Gemini(id=MODEL_NAMES[model_class])
        case _:
            raise ValueError(f"Model name {model_class} not supported")


class MujocoRobotTools(Toolkit):
    def __init__(self, robot: MujocoRobot, **kwargs):
        self.robot = robot
        super().__init__(
            name="mujoco_robot",
            tools=[self.set_robot_state_and_render],
            **kwargs,
        )

    def set_robot_state_and_render(
        self,
        state: list[float],
    ) -> ToolResult:  # see https://github.com/agno-agi/agno/issues/4436
        """Set the robot joint positions and return the rendered images."""
        try:
            log_debug(f"Setting robot state to {state}")
            self.robot.set_state(state)
            images = self.robot.render()
            pil_image = get_images_as_grid(images)
            mcp_image = pil_to_mcp_image(pil_image)
            image_artifact = ImageArtifact(
                id=str(uuid4()),
                content=mcp_image.data,
                mime_type=mcp_image._mime_type,
            )
            return ToolResult(
                content="Robot state set successfully",
                images=[image_artifact],
            )
        except Exception as e:
            logger.error(f"Error setting robot state to {state}: {e}")
            return f"Error: {e}"


class JointConfigurationAgent:
    def __init__(
        self,
        robot: MujocoRobot,
        model_class: ModelClass,
    ):
        model = get_model(model_class)
        instructions = get_achieve_pose_prompt(robot)
        tools = MujocoRobotTools(robot)
        self.agent = Agent(
            model=model,
            tools=[tools],
            instructions=instructions,
            telemetry=False,
        )

    @weave.op()
    def get_joint_positions(self, reference_image_paths: list[str]) -> list[float]:
        images = [Image(filepath=image_path) for image_path in reference_image_paths]
        response: RunOutput = self.agent.run(
            "Match the joint positions to the reference images",
            images=images,
        )
        assert len(response.tools) > 1, (
            "The agent should have used the tool at least once"
        )
        # the final joint configuration is the argument of the last tool call
        return response.tools[-1].tool_args["state"]


class GroundTruthSource(StrEnum):
    REAL = "real"
    SIMULATED = "simulated"


def get_ground_truth(
    ground_truth_source: GroundTruthSource, ground_truth_seed: int
) -> tuple[list[str], list[float]]:
    match ground_truth_source:
        case GroundTruthSource.REAL:
            return get_real_ground_truth(ground_truth_seed)
        case GroundTruthSource.SIMULATED:
            return get_simulated_ground_truth(ground_truth_seed)
        case _:
            raise ValueError(f"Ground truth source {ground_truth_source} not supported")


def get_real_ground_truth(ground_truth_seed: int) -> tuple[list[str], list[float]]:
    # use lerobot
    pass


def get_simulated_ground_truth(
    robot: MujocoRobot, ground_truth_seed: int
) -> tuple[list[str], list[float]]:
    # sample a random joint configuration with the given seed
    np.random.seed(ground_truth_seed)

    ground_truth_joint = np.random.uniform(
        *robot._joint_ranges,
    )

    robot.set_state(ground_truth_joint)
    images = robot.render()
    number_of_reference_images = 2

    # Create temporary directory for saving reference images
    temp_dir = Path(tempfile.mkdtemp())
    image_paths = []

    # Save the required number of reference images
    for i in range(min(number_of_reference_images, len(images))):
        # Convert numpy array to PIL Image
        pil_image = PILImage.fromarray(images[i])

        # Create unique filename with seed and index
        filename = f"reference_{ground_truth_seed}_{i}.png"
        image_path = temp_dir / filename

        # Save image to temporary file
        pil_image.save(image_path)
        image_paths.append(str(image_path))

    return image_paths, ground_truth_joint


def end_effector_pose_distance(
    robot: MujocoRobot,
    ground_truth_joint_positions: list[float],
    predicted_joint_positions: list[float],
) -> float:
    ground_truth_end_effector_pose = get_end_effector_pose(
        robot, ground_truth_joint_positions
    )
    predicted_end_effector_pose = get_end_effector_pose(
        robot,
        predicted_joint_positions,
    )
    return np.linalg.norm(ground_truth_end_effector_pose - predicted_end_effector_pose)


def get_end_effector_pose(
    robot: MujocoRobot,
    joint_positions: list[float],
) -> np.ndarray:
    if robot.config.robot_name != "so_arm100_mj_description":
        raise ValueError("Only so_arm100_mj_description is supported for now")
    end_effector_body_name = "Fixed_Jaw"
    robot.set_state(joint_positions)
    position = robot._data.body(end_effector_body_name).xpos.copy()
    quaternion = robot._data.body(end_effector_body_name).xquat.copy()
    # TODO: these two are not necessarily compatible units
    return np.concatenate([position, quaternion])


def main(
    robot_name: str,
    model_class: ModelClass,
    ground_truth_source: GroundTruthSource,
    ground_truth_seed: int,
    # ideally we would set a seed parameter for both gemini and claude
):
    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        config={
            "robot_name": robot_name,
            "model_class": model_class,
            "ground_truth_source": ground_truth_source,
            "ground_truth_seed": ground_truth_seed,
        },
    )
    robot = MujocoRobot.from_robot_name(robot_name)
    if ground_truth_source == GroundTruthSource.SIMULATED:
        image_paths, ground_truth_joint_positions = get_simulated_ground_truth(
            robot,
            ground_truth_seed,
        )
    else:
        raise NotImplementedError("Real ground truth is not implemented yet")
    agent = JointConfigurationAgent(robot, model_class)
    predicted_joint_positions = agent.get_joint_positions(image_paths)

    metric_value = end_effector_pose_distance(
        robot,
        ground_truth_joint_positions,
        predicted_joint_positions,
    )
    run.log({"end_effector_pose_distance": metric_value})
    run.finish()


if __name__ == "__main__":
    typer.run(main)
