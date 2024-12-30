from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq Model
groq_model = Groq(id="llama3-groq-70b-8192-tool-use-preview")

# Web Search Agent
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information in the response."],
    show_tool_calls=True,
    markdown=True,
)

# Financial Agent
financial_agent = Agent(
    name="Financial Agent",
    role="Get financial data",
    model=groq_model,
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use Tables to display the data."],
    show_tool_calls=True,
    markdown=True,
)

# Multi-Agent Setup
multi_ai_agent = Agent(
    team=[websearch_agent, financial_agent],
    model=groq_model,
    instructions=[
        "Always include the source of the information in the response.",
        "Use Tables to display the data.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Test the Agent
multi_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA.", stream=True
)
