from langchain.chat_models import ChatOpenAI
from vllm import SamplingParams

# Define vLLM sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

# Convert manually
openai_params = {
    "temperature": sampling_params.temperature,
    "top_p": sampling_params.top_p,
    "max_tokens": sampling_params.max_tokens,
}

# Initialize ChatOpenAI with the converted parameters
chat_model = ChatOpenAI(model_name="your-vllm-model", openai_api_base="http://localhost:8000", **openai_params)

response = chat_model.predict("What is self-refinement in LLMs?")
print(response)
