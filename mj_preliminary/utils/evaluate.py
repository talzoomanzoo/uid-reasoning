from fuzzywuzzy import fuzz
from utils.config import step_level_evaluation_config
from langchain_core.messages import SystemMessage, HumanMessage

def evaluate_prediction(prediction: str, answer: str):
    if prediction == answer:
        return True
    else:
        similarity_score = fuzz.ratio(prediction, answer) #fuzzy matching for parsing error
        threshold = 80
        if similarity_score >= threshold:
            return True
        else:
            return False

async def step_level_evaluate(raw_response: str, model_name: str):
    llm, eval_prompt = step_level_evaluation_config(model_name=model_name)
    messages = [[SystemMessage(content=eval_prompt), HumanMessage(content=raw_response)]] #reconfigure
    response = await llm.agenerate(messages)
    return response.generations[0][0].text

def step_final_correlation(raw_response: str):
    pass