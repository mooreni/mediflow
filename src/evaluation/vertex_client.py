# Vertex AI client setup for MediFlow.
# Provides three client factories:
#   get_client()           — shared singleton for single-threaded / main-thread use
#   get_eval_client()      — thread-local client for concurrent evaluation workers
#   get_translate_client() — thread-local client for concurrent translation workers
#
# genai.Client is NOT thread-safe: its internal httpx.AsyncClient is created
# eagerly and must not be shared across threads that call asyncio.run()
# concurrently.  Always use a thread-local factory inside ThreadPoolExecutor.

import os
import threading

from google import genai

_client: genai.Client | None = None
_thread_local = threading.local()


def get_client() -> genai.Client:
    """Initialize the Google GenAI SDK with Vertex AI and return a client.

    Reads GOOGLE_CLOUD_PROJECT (required) and GOOGLE_CLOUD_LOCATION (optional,
    defaults to global) from environment variables. Reuses a cached
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


def get_eval_client() -> genai.Client:
    """Return a thread-local GenAI client for async evaluation calls.

    Each thread gets its own genai.Client instance because the underlying
    httpx.AsyncClient (created eagerly in genai.Client.__init__) is not
    thread-safe. When multiple ThreadPoolExecutor workers each call
    asyncio.run() concurrently, each needs an isolated async HTTP
    connection pool.

    Returns:
        A genai.Client bound to the current thread.

    Raises:
        KeyError: If GOOGLE_CLOUD_PROJECT is not set.
    """
    client = getattr(_thread_local, "eval_client", None)
    if client is not None:
        return client
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    _thread_local.eval_client = genai.Client(
        vertexai=True, project=project, location=location
    )
    return _thread_local.eval_client


def get_translate_client() -> genai.Client:
    """Return a thread-local GenAI client for concurrent translation calls.

    Each ThreadPoolExecutor worker thread gets its own genai.Client so that
    the underlying httpx connection pool is never shared across threads.
    Without isolation, concurrent generate_content calls on a single shared
    client exhaust the pool and trigger read-timeout errors.

    Returns:
        A genai.Client bound to the current thread.

    Raises:
        KeyError: If GOOGLE_CLOUD_PROJECT is not set.
    """
    client = getattr(_thread_local, "translate_client", None)
    if client is not None:
        return client
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    _thread_local.translate_client = genai.Client(
        vertexai=True, project=project, location=location
    )
    return _thread_local.translate_client


MODEL = "gemini-3.1-pro-preview"