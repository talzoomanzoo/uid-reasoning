
def get_task_instruction_math(question, model_name=None):
    user_prompt = (
            'Please answer the following math question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following multiple-choice question. '
            'Your choice should ONLY be one of the letters A, B, C, or D. DO NOT include any answer content.\n\n'
            'Always should provide your choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Your choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
            'Provide your choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following multiple-choice question.\n\n'
            'Your choice should be one of the letters A, B, C, or D. DO NOT include any answer content.\n\n'
            'Always provide your choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt