def sigmoid_prime(s):
    return s * (1 - s)

def backpropagation(l1, l2, weights, y):
    # Zuerst berechnen der Fehler an der Ausgangsschicht in dem Beispiel gab es zwei Schichten
    l2_error = y.reshape(-1, 1) - l2
    # Multiplizieren der Fehler mit der Ableitungs der Aktivierungsfunktion das Ergebnis ist ein Gradient
    l2_delta = l2_error * sigmoid_prime(l2)
    l1_error = l2_delta.dot(weights[1].T)
    # Eine weitere Gradientenberechnung liefert die notwendigen Korrekturen für die Eingabeschicht
    l1_delta = l1_error * sigmoid_prime(l1)
    # Das ist eine Prozedur für ein Netz bestehend aus einer Eingabeschicht, einer verborgenen Schicht und einer Ausgabeschicht