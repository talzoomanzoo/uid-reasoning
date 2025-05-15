import torch

def format_prompt_response(prompt: str, response: str) -> str:
    instruction_template = "<|begin_of_sentence|><|User|>\n"
    response_template = "<|Assistant|><think>\n"
    return f"{instruction_template}\n{prompt}\n<|im_end|>\n{response_template}{response}<|im_end|>"

def score_response(model, tokenizer, prompt: str, response: str) -> float:
    full_text = format_prompt_response(prompt, response)
    tokenized = tokenizer(full_text, return_tensors="pt")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    # Identify where the assistant's response starts
    response_start_str = "<|Assistant|><think>\n"
    response_start_idx = full_text.find(response_start_str)
    if response_start_idx == -1:
        raise ValueError("Could not find response start token.")

    response_token_start = tokenizer(full_text[:response_start_idx + len(response_start_str)], return_tensors="pt").input_ids.shape[1]

    # Mask out tokens before assistant response
    labels = input_ids.clone()
    labels[:, :response_token_start] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss.item()

# def get_preference(model, tokenizer, output_list: list, labeled_answer: str, num_responses: int):
#     scores = [score_response(model, tokenizer, labeled_answer, resp) for resp in output_list]

#     ranked = sorted(zip(output_list, scores), key=lambda x: x[1])
#     chosen_response, chosen_loss = ranked[0] #lowest loss
#     rejected_response, rejected_loss = ranked[-1] #highest loss

#     return {
#         "chosen": chosen_response,
#         "rejected": rejected_response,
#         "loss_chosen": chosen_loss,
#         "loss_rejected": rejected_loss,
#         "all_losses": ranked
#     }

