import importlib
from typing import Annotated

from pydantic import AfterValidator
from pydantic_settings import BaseSettings, SettingsConfigDict
from robot_descriptions import DESCRIPTIONS


def is_a_valid_mjcf_description(description: str) -> str:
    description_ = DESCRIPTIONS.get(description)
    if description_ is None or not description_.has_mjcf:
        raise ValueError(f"Invalid robot description: {description}")
    return description


class Config(BaseSettings):
    robot_name: Annotated[str, AfterValidator(is_a_valid_mjcf_description)] = (
        "so_arm100_mj_description"
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MUJOCO_MCP_")

    @property
    def robot_mjcf_path(self) -> str:
        # import self.robot_name from robot_descriptions
        robot_module = importlib.import_module(f"robot_descriptions.{self.robot_name}")
        return robot_module.MJCF_PATH


config = Config()
