import os
import argparse
import requests
from dotenv import load_dotenv

API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-01-preview')


def diagnose(env_path: str | None):
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded env: {env_path}")
    else:
        default_env = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(default_env):
            load_dotenv(default_env)
            print(f"Loaded default env: {default_env}")
        else:
            print("No .env found. Provide --env path.")
            return 2

    key = os.getenv('AZURE_OPENAI_API_KEY')
    resource = os.getenv('AZURE_OPENAI_API_RESOURCE')
    if not key:
        print("Missing AZURE_OPENAI_API_KEY")
        return 2
    if not resource:
        print("Missing AZURE_OPENAI_API_RESOURCE")
        return 2

    url = f"https://{resource}.openai.azure.com/openai/deployments?api-version={API_VERSION}"
    headers = {"api-key": key}
    try:
        r = requests.get(url, headers=headers)
    except requests.exceptions.RequestException as e:
        print(f"Network error reaching resource '{resource}': {e}")
        return 3

    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        try:
            data = r.json()
        except Exception:
            print("Got non-JSON response from Azure.")
            return 4
        deployments = [d.get('model') or d.get('name') for d in data.get('data', [])]
        print(f"Deployments found: {deployments}")
        if not deployments:
            print("No deployments returned; key+resource valid, but no deployments configured.")
        print("Conclusion: Resource and API key look valid (200).")
        return 0
    elif r.status_code == 401:
        print("Conclusion: API key invalid for this resource (401 Unauthorized).")
        return 1
    elif r.status_code == 404:
        print("Conclusion: Resource name likely incorrect or not found (404).")
        return 1
    else:
        print(f"Unexpected response: {r.text[:200]}")
        return 5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose Azure OpenAI resource/key by listing deployments')
    parser.add_argument('--env', type=str, help='Path to .env file')
    args = parser.parse_args()
    code = diagnose(args.env)
    raise SystemExit(code)
