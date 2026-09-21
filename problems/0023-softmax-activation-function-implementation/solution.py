import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores=np.array(scores)
    scores=scores-np.max(scores)
    scores=np.exp(scores)
    scores=scores/np.sum(scores)

    return scores.tolist()