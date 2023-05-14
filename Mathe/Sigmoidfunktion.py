import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Zum Berechnen der Ableitung der Sigmoidfunktion 

def sigmoid_prime(s):
    return s * (1 -s)