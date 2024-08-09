from dotenv import load_dotenv
from langsmith import Client
import os

def load_and_configure_environment():
    """Load environment variables and set configurations."""
    load_dotenv(override=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGSMITH_API_KEY"))
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "multi-agent-storywriter"

def get_env_variable(var_name: str, required: bool = True) -> str:
        """Helper function to retrieve environment variables and raise error if missing."""
        value = os.getenv(var_name)
        if required and not value:
            raise ValueError(f"Required environment variable {var_name} is missing")
        return value