'''
Die Aktivierungsfunktion sollte definiert worden sein.
Die vorwärts gerichtitete Prozedur funktioniert durch die Matrixmultiplikation von der Eingabe einer jeden
Schicht mit den Verbindungsgewichten. 
Das folgende Codesegment integrierte die Sigmoidfunktion in die Feedforward-Funktionalität des Netzes.
'''
import numpy as np
def sigmoid(z):
    return 1/(1 + np.exp(-z))


def feed_forward(X, weights):
    a = X.copy()
    out = list()
    for W in weights:
        z = np.dot(a, W)
        a = sigmoid(z)
        out.append(a)
    return out