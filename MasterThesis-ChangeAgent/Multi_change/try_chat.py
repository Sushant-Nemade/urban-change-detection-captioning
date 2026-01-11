from lagent.agents import ReAct
from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter, Visual_Change_Process_PythonInterpreter
from lagent.llms import GPTAPI
import os
from dotenv import load_dotenv
import pathlib

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# Load .env: prefer provided Azure path, fallback to local .env
preferred_env = r"C:\ChangeAgentMCI2\Multi_change\.env"
local_env = str(pathlib.Path(__file__).with_name('.env'))
if os.path.exists(preferred_env):
    load_dotenv(preferred_env)
elif os.path.exists(local_env):
    load_dotenv(local_env)

openai_key = os.getenv('OPENAI_API_KEY')
azure_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_resource = os.getenv('AZURE_OPENAI_API_RESOURCE')
azure_deployment = os.getenv('AZURE_OPENAI_API_DEPLOYMENT')

azure_env_present = all([azure_key, azure_resource, azure_deployment])
if azure_env_present:
    llm = GPTAPI(model_type='gpt-4o-mini', key='ENV', azure_mode=True)
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
chatbot = ReAct(llm=llm, action_executor=ActionExecutor(actions=actions))

if __name__ == '__main__':
    history = []
    while True:
        user_input = input('\n||<User>||Please input your message: ')
        # user_input='The path of picture A is F:/LCY/Change_Agent/Multi_change/predict_result/test_000004_A.png, the path of picture B is F:/LCY/Change_Agent/Multi_change/predict_result/test_000004_B.png, the masks for detecting changes in buildings and roads are displayed in red and green respectively. The result can be saved in E:/1.png'
        # user_input='The path of picture A is F:/LCY/Change_Agent/Multi_change/predict_result/test_000004_A.png, the path of picture B is F:/LCY/Change_Agent/Multi_change/predict_result/test_000004_B.png, Describe what has changed in the two images'

        print('\n||Start thinking ...')
        history.append(dict(role='user', content=user_input))
        agent_return = chatbot.chat(history)
        history.append(dict(role='assistant', content=agent_return.response))
        print('\n||<Change-Agent>|| The response of Agent:')
        print(agent_return.response)
