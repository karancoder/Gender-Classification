import numpy as np

class Classifier(object):
    """
    randomly initialize weights
    """
    def __init__(self, dim, filename):
        self.dim = dim
        self.weights = np.load(filename)

    """
    evaluate the probability of belonging to class 1
    P(C|phi_n) = sigmoid(w.T * phi_n)
    """
    def evaluate(self,phi_n):
        return sig(np.dot(self.weights, phi_n.T))

    """
    predict test data: return 1 if the datum is more probable to be in class 1 
    and 0 if the datum is more probable to be in class 0
    round to the closest integer
    """
    def predict(self,phi_n):
        return int(round(self.evaluate(phi_n)[0]))
        
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))