def average_score(scores: list[str]):
    return sum(1 for score in scores if score == True) / len(scores)