# Connection health check for Vertex AI + Gemini 3.1 Pro.
# Run this first to verify gcloud credentials and project configuration are working.
# Usage: python3 scripts/test_connection.py

import os
import sys

from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    print("ERROR: GOOGLE_CLOUD_PROJECT environment variable is not set.")
    print("Set it in your shell or in a .env file, then retry.")
    sys.exit(1)

from src.evaluation.vertex_client import MODEL, get_client

print(f"Project : {os.environ['GOOGLE_CLOUD_PROJECT']}")
print(f"Location: {os.environ.get('GOOGLE_CLOUD_LOCATION', 'global')}")
print(f"Model   : {MODEL}")
print("Initializing client...")

try:
    client = get_client()
    print("Sending test prompt...")
    response = client.models.generate_content(
        model=MODEL,
        contents="Reply with exactly: OK",
    )
    reply = response.text.strip()
    print(f"Response: {reply}")
    if "OK" in reply:
        print("\nConnection successful.")
        sys.exit(0)
    else:
        print("\nWARNING: Unexpected response (expected 'OK').")
        sys.exit(0)
except Exception as exc:
    print(f"\nERROR: {exc}")
    sys.exit(1)
