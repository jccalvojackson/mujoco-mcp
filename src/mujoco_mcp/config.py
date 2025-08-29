from typing import Annotated

from pydantic import AfterValidator
from pydantic_settings import BaseSettings, SettingsConfigDict
from robot_descriptions import DESCRIPTIONS


def is_a_valid_mjcf_description(description: str) -> bool:
    description_ = DESCRIPTIONS.get(description)
    if description_ is None:
        return False
    return description_.has_mjcf


class Config(BaseSettings):
    robot_name: Annotated[str, AfterValidator(is_a_valid_mjcf_description)] = (
        "so_arm100_mj_description"
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MUJOCO_MCP_")


config = Config()
