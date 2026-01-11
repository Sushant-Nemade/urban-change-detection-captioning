from lagent.agents import ReAct
from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter, Visual_Change_Process_PythonInterpreter
from lagent.llms import GPTAPI
import os
from dotenv import load_dotenv
import pathlib
import argparse

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

def build_chatbot(env_path: str | None = None) -> ReAct:
    # Load .env: use provided path, else default preferred, else local .env
    preferred_env = r"C:\ChangeAgentMCI2\Multi_change\.env"
    local_env = str(pathlib.Path(__file__).with_name('.env'))
    selected_env = None
    if env_path and os.path.exists(env_path):
        selected_env = env_path
    elif os.path.exists(preferred_env):
        selected_env = preferred_env
    elif os.path.exists(local_env):
        selected_env = local_env
    if selected_env:
        load_dotenv(selected_env)
        print(f"Loaded environment from: {selected_env}")

    openai_key = os.getenv('OPENAI_API_KEY')
    azure_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_resource = os.getenv('AZURE_OPENAI_API_RESOURCE')
    azure_deployment = os.getenv('AZURE_OPENAI_API_DEPLOYMENT')

    azure_env_present = all([azure_key, azure_resource, azure_deployment])
    if azure_env_present:
        llm = GPTAPI(model_type='gpt-5', key='ENV', azure_mode=True)
    else:
        if not openai_key:
            print('Warning: OPENAI_API_KEY not set and Azure vars missing. LLM responses will not work without credentials.')
        llm = GPTAPI(model_type='gpt-4o-mini', key=openai_key)

    google_key = os.getenv('GOOGLE_API_KEY')
    search_tool = GoogleSearch(api_key=google_key) if google_key else None
    python_interpreter = PythonInterpreter()
    imgchange_python_interpreter = Visual_Change_Process_PythonInterpreter()

    actions = [imgchange_python_interpreter]
    if search_tool:
        actions.insert(0, search_tool)
    return ReAct(llm=llm, action_executor=ActionExecutor(actions=actions))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Change-Agent chat CLI')
    parser.add_argument('--env', type=str, help='Path to .env file with Azure/OpenAI credentials')
    args = parser.parse_args()

    chatbot = build_chatbot(args.env)
    history = []
    while True:
        user_input = input('\n||<User>||Please input your message: ')
        print('\n||Start thinking ...')
        history.append(dict(role='user', content=user_input))
        agent_return = chatbot.chat(history)
        history.append(dict(role='assistant', content=agent_return.response))
        print('\n||<Change-Agent>|| The response of Agent:')
        print(agent_return.response)
