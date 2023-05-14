import numpy as np

def doAdd(a, b):
    return a + b
vectAdd = np.vectorize(doAdd)
print(vectAdd([1, 2, 3, 4], [1, 2, 3, 4]))
# [2 4 6 8]

'''
Zunächst wird die Funktion doAdd definiert, die zwei Skalare a und b entgegennimmt und in der nächsten Codezeile addiert
und die Summe zurückgibt.
Dann wird die Funktion vectAdd definiert.
'''