import getpass
import os
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_core.prompts import PromptTemplate
from config import LLM_MODEL, LLM_NAME, LLM_PROVIDER, ENV_VAR_NAME, API_KEY

def init_llm():
    key = API_KEY or getpass.getpass(f"Enter API key for {LLM_NAME}: ")
    os.environ[ENV_VAR_NAME] = key
    return init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER)

def load_prompt(template = "rlm/rag-prompt"):
    custom_prompt = PromptTemplate.from_template(template)
    return custom_prompt
