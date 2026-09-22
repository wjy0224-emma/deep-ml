import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	p=np.array(P)
	q=np.array(Q)
	mask=p>0

	entropy=-np.sum(p[mask]*np.log(p[mask]))
	cross_entropy=-np.sum(p[mask]*np.log(q[mask]))

	return entropy,cross_entropy