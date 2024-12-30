import os

import phi
import phi.api
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.playground import Playground, serve_playground_app
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

load_dotenv()

# initialize phi api
phi.api = os.getenv("PHI_API_KEY")

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

app = Playground(agents=[financial_agent, websearch_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
