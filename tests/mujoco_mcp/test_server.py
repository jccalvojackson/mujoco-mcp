import pytest
from mcp.shared.memory import (
    create_connected_server_and_client_session as create_session,
)
from mcp.types import CallToolResult, ImageContent

from mujoco_mcp.server import mujoco_mcp_server, robot


@pytest.mark.anyio
async def test_list_tools():
    async with create_session(mujoco_mcp_server._mcp_server) as client_session:
        # Test without cursor parameter (omitted)
        EXPECTED_NUMBER_OF_TOOLS = 1
        result = await client_session.list_tools()
        assert len(result.tools) == EXPECTED_NUMBER_OF_TOOLS

        expected_tool_name = "set_robot_state"
        assert result.tools[0].name == expected_tool_name

        joint_bounds = robot.joint_bounds
        assert str(joint_bounds) in result.tools[0].description


@pytest.mark.anyio
async def test_call_set_robot_state():
    async with create_session(mujoco_mcp_server._mcp_server) as client_session:
        result: CallToolResult = await client_session.call_tool(
            "set_robot_state",
            arguments={"state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        )
        assert len(result.content) == 1
        assert isinstance(result.content[0], ImageContent)
        expected_mime_type = "image/webp"
        assert result.content[0].mimeType == expected_mime_type


@pytest.mark.anyio
async def test_list_prompts():
    async with create_session(mujoco_mcp_server._mcp_server) as client_session:
        result = await client_session.list_prompts()
        assert len(result.prompts) == 1
