# Vertex AI client setup for MediFlow evaluation.
# Initializes the Google GenAI SDK with Vertex AI backend and returns a
# configured Gemini 3.1 Pro client. All evaluation modules import get_client()
# from here so initialization happens once.

import os

from google import genai

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Initialize the Google GenAI SDK with Vertex AI and return a client.

    Reads GOOGLE_CLOUD_PROJECT (required) and GOOGLE_CLOUD_LOCATION (optional,
    defaults to us-central1) from environment variables. Reuses a cached
    instance if already initialized.

    Returns:
        A configured genai.Client instance pointed at Vertex AI.

    Raises:
        KeyError: If GOOGLE_CLOUD_PROJECT is not set.
    """
    global _client
    if _client is not None:
        return _client

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    _client = genai.Client(vertexai=True, project=project, location=location)
    return _client


MODEL = "gemini-3.1-pro-preview"
