import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq Model
groq_model = Groq(id="llama3-groq-8b-8192-tool-use-preview")

# Web Search Agent
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information in the response."],
    # show_tool_calls=True,
    # markdown=True,
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
    # show_tool_calls=True,
    # markdown=True,
)

# Multi-Agent Setup
multi_ai_agent = Agent(
    team=[financial_agent, websearch_agent],
    model=groq_model,
    instructions=[
        "Always include the source of the information in the response.",
        "Use Tables to display the data.",
    ],
    # show_tool_calls=True,
    # markdown=True,
)

st.title("Financial Agent")

user_query = st.text_input("Ask the chatbot a question:", "")

if user_query:
    with st.spinner("Fetching response..."):
        response = multi_ai_agent.run(user_query)
        # print(response.content)
        st.markdown(response.content)
