import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	arr=np.array(scores,dtype=float)
	arr=arr-arr.max()
	log_sum_exp=np.log(np.sum(np.exp(arr)))
	return arr-log_sum_exp