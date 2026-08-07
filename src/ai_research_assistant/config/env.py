import os
from dotenv import load_dotenv

load_dotenv()

class EnvConfig:
    """
    Loads and validates environment variables.
    """

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    @classmethod
    def validate(cls, required_keys: list[str] = None):
        """
        Validate required environment variables.        
        """

        if required_keys is None:
            required_keys = []

        missing_keys = [
            key for key in required_keys
            if not getattr(cls, key, None)
        ]

        if missing_keys:
            raise EnvironmentError(
                "Missing required environment variables:\n"
                + "\n".join(f"- {key}" for key in missing_keys)
            )