import asyncio

from agno.agent import Agent
from agno.media import Image
from agno.models.anthropic import Claude
from agno.playground import Playground
from agno.tools.mcp import MCPTools
from agno.tools.reasoning import ReasoningTools
from mcp.client.session import ClientSession
from mcp.types import GetPromptResult, ListPromptsRequest

MCP_COMMAND = "uv run --directory /Users/jccj/personal/git_projects/mujoco-mcp/ /Users/jccj/personal/git_projects/mujoco-mcp/src/mujoco_mcp/server.py"

simulation_mcp = MCPTools(MCP_COMMAND)


async def _get_prompt(client_session: ClientSession) -> str:
    prompts: ListPromptsRequest = await client_session.list_prompts()
    assert len(prompts.prompts) == 1
    prompt_name: str = prompts.prompts[0].name
    prompt: GetPromptResult = await client_session.get_prompt(prompt_name)
    assert len(prompt.messages) == 1
    prompt_message: str = prompt.messages[0].content.text
    return prompt_message


async def _get_agent() -> Agent:
    await simulation_mcp.connect()
    prompt: str = await _get_prompt(simulation_mcp.session)
    agent = Agent(
        model=Claude(id="claude-sonnet-4-20250514"),
        tools=[
            ReasoningTools(add_instructions=True),
            simulation_mcp,
        ],
        instructions=prompt,
        telemetry=False,
    )
    return agent


async def _run_calibration_agent(agent: Agent, reference_images: list[Image]):
    await agent.aprint_response(
        images=reference_images,
    )

    await simulation_mcp.close()


def get_agent():
    return asyncio.run(_get_agent())


agent = get_agent()
playground = Playground(agents=[agent])

app = playground.get_app()

if __name__ == "__main__":
    from pathlib import Path

    image_directory = Path(__file__).parent.parent / "assets"
    image_path_names = ["test_15efc020ad_0.png", "test_b59c1bd94b_0.png"]
    images = [
        Image(filepath=image_directory / image_path_name)
        for image_path_name in image_path_names
    ]

    playground.serve(app)

    # Uncomment to run calibration instead of playground
    # await _run_calibration_agent(agent, images)
