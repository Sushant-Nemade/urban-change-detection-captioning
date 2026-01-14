from lagent.agents import ReAct
from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter, Visual_Change_Process_PythonInterpreter
from lagent.llms import GPTAPI
import os
from dotenv import load_dotenv
import argparse

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

def build_chatbot(env_path: str) -> ReAct:
    if not env_path or not os.path.exists(env_path):
        raise RuntimeError(f"Missing or invalid --env path: {env_path!r}")
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")

    openai_key = os.getenv('OPENAI_API_KEY')
    azure_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_resource = os.getenv('AZURE_OPENAI_API_RESOURCE')
    azure_deployment = os.getenv('AZURE_OPENAI_API_DEPLOYMENT')

    azure_env_present = all([azure_key, azure_resource, azure_deployment])
    if azure_env_present:
        llm = GPTAPI(model_type='gpt-5', key='ENV', azure_mode=True)
    else:
        if not openai_key:
            missing = []
            if not azure_key:
                missing.append('AZURE_OPENAI_API_KEY')
            if not azure_resource:
                missing.append('AZURE_OPENAI_API_RESOURCE')
            if not azure_deployment:
                missing.append('AZURE_OPENAI_API_DEPLOYMENT')
            missing.append('OPENAI_API_KEY')
            raise RuntimeError(
                'Missing credentials. Provide either OpenAI or Azure OpenAI:\n'
                f'  Missing vars: {", ".join(missing)}\n'
                'Expected one of:\n'
                '  - OPENAI_API_KEY\n'
                '  - AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_RESOURCE, AZURE_OPENAI_API_DEPLOYMENT'
            )
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
    parser.add_argument('--env', type=str, required=True, help='Path to .env file with Azure/OpenAI credentials')
    parser.add_argument('--once', type=str, help='Run a single prompt and exit')
    args = parser.parse_args()

    chatbot = build_chatbot(args.env)
    history = []
    if args.once:
        user_input = args.once
        print('\n||<User>|| One-shot message:')
        print(user_input)
        print('\n||Start thinking ...')
        history.append(dict(role='user', content=user_input))
        agent_return = chatbot.chat(history)
        print('\n||<Change-Agent>|| The response of Agent:')
        print(agent_return.response)
    else:
        while True:
            user_input = input('\n||<User>||Please input your message: ')
            print('\n||Start thinking ...')
            history.append(dict(role='user', content=user_input))
            agent_return = chatbot.chat(history)
            history.append(dict(role='assistant', content=agent_return.response))
            print('\n||<Change-Agent>|| The response of Agent:')
            print(agent_return.response)
