import numpy as np
# Bedeutung von Tensoren für Deep Learning
# Man solle sich die arbeit mit einem Algorithmus vorstellen, der drei Eingaben benögtigt dargestellt durch einen Vektor
inputs = np.array([5, 10, 15])

# Es handelt sich um Einzelwerte basierend auf einem einzigen Ergebnis.
# Dieses könnte zum Beispiel eine Suchanfrage sein, um herauszufinden, welches Waschmittel am besten geeignet ist.
# Bevor diese Werte dem Algorithmus zugeführt werden, müssen diese Werte entsprechend gewichtet werden.
# Diese Gewichte können durch folgende Werte dargestellt werden:
weights = np.array([[.5, .2, -1], [.3, .4, .1], [-.2, .1, .3]])
# Jetzt können die Eingaben mit den Gewichten transformiert werden.
# Der Algorithmus verwendet sozusagen das Gelehrnte auf neue Daten an.
result = np.dot(inputs, weights)
print(result)
# Ausgabe: [2.5 6.5 0.5]
# Der Vektor inputs ist eine verborgene Schicht in einem neuronalen Netzwerk und die Ausgabe result ist die nächste
# verborgene Schicht in denselben Netz
