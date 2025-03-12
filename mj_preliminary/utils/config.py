from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm_asyncio
import os
import asyncio
import json

def step_level_evaluation_config(model_name: str):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    with open("./prompts/step_level_evaluate.json", "r") as f:
        eval_prompt = json.load(f)
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=3000,
        max_retries=100,
        openai_api_key=openai_api_key
    )
    return llm, eval_prompt
    
# def load_and_prepare_step_level_evaluate_data(raw_response: str):
#     with open("./prompts/step_level_evaluate.json", "r") as f:
#         prompt_data = json.load(f)
#     return prompt_data["user_input"].format(**{
#         "raw_response": raw_response
#         })
    