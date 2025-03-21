def average_score(scores: list[str]):
    return sum(1 for score in scores if score == True) / len(scores)

def average_token_used(current_token_used: int, current_num: int):
    return current_token_used / current_num