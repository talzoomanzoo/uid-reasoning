from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

def duck_search(query: str, model_name: str = "gpt-4o"):
    

    agent = Agent(
        model=OpenAIChat(id=model_name),
        description="You need to retrieve information from the web to answer the question.",
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )
    return agent.print_response(query, stream=True)