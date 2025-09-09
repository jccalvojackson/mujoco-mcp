import numpy as np
import typer

import wandb
from mujoco_mcp.server import robot

from .evaluate_directly import end_effector_pose_distance

WANDB_PROJECT_NAME = "evaluate_joint_configuration_ai_agent"


def main(seed: int):
    np.random.seed(seed)
    ground_truth_joint = np.random.uniform(
        *robot._joint_ranges,
    )
    random_joint = np.random.uniform(
        *robot._joint_ranges,
    )
    distance = end_effector_pose_distance(
        robot,
        ground_truth_joint,
        random_joint,
    )
    wandb_settings = wandb.Settings(console="off")
    with wandb.init(
        settings=wandb_settings,
        project=WANDB_PROJECT_NAME,
        config={
            "ground_truth_seed": seed,
            "robot_name": robot.config.robot_name,
            "model_class": "random",
            "ground_truth_source": "simulated",
        },
    ) as run:
        run.log({"end_effector_pose_distance": distance})


if __name__ == "__main__":
    typer.run(main)
