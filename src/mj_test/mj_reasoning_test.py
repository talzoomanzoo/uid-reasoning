from openai import OpenAI
from pydantic import BaseModel
import os

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = "http://localhost:8100/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "talzoomanzoo/DeepSeek-R1-Distill-Qwen-14B-awq"


class Answer(BaseModel):
    answer: str


json_schema = Answer.model_json_schema()

prompt = ("Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$")
completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)