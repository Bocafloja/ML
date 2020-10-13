import csv
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import sklearn.metrics
import pylab

# Importando los datos
with open('Bias_correction_ucl.csv', mode='r', encoding='utf8') as DS:
    lector = csv.reader(DS, delimiter=',')
    DataSet = []
    for datos in lector:
        DataSet.append(datos)
DataSet = DataSet[1:]
DataFrameY = np.array([dato[22] for dato in DataSet])
DataFrameX = np.array([dato[:21] for dato in DataSet])
DataSet = (DataFrameX, DataFrameY)

# Parámetros de la red
units_capa1 = 22
units_capa2 = 1
w1 =npr.rand(len(DataSet[0]), units_capa1)
b1 = npr.rand(units_capa1)
w2 = npr.rand(units_capa1, units_capa2)
b2 = 0.0
theta = (w1, b1, w2, b2)

# Función de costo
def costo_cuadratico(y, y_barra):
    return np.dot((y - y_barra), (y - y_barra))

# Armando la red
def red_neuronal(x, theta):
    w1, b1, w2, b2 = theta
    return np.tanh(np.dot((np.tanh(np.dot(x, w1) + b1)), w2) + b2)

# Armando la función objetivo a ser optimizada
def objetivo(theta, idx):
    return costo_cuadratico(DataSet[1][idx], red_neuronal(DataSet[0][idx], theta))

# Actualizar theta
def actualizar_theta(theta, delta, alpha):
    w1, b1, w2, b2 = theta
    w1_delta, b1_delta, w2_delta, b2_delta = delta
    w1_nuevo = w1 - alpha * w1_delta
    b1_nuevo = b1 - alpha * b1_delta
    w2_nuevo = w2 - alpha * w2_delta
    b2_nuevo = b2 - alpha * b2_delta
    nuevo_theta = (w1_nuevo, b1_nuevo, w2_nuevo, b2_nuevo)
    return nuevo_theta

# Calculando el gradiente
grad_objetivo = grad(objetivo)

# Entrenando la red neuronal

#print("RMSE antes de entrenar: ", sklearn.metrics.mean_squared_error(DataSet[1], red_neuronal(DataSet[0], theta)))
rmse = []
for i in range(len(DataSet[0])):
    for j in range(len(DataSet)):
        delta = grad_objetivo(theta, j)
        theta = actualizar_theta(theta, delta, 0.01)
        rmse.append(sklearn.metrics.mean_squared_error(DataSet[1], red_neuronal(DataSet[0], theta)))
print("RMSE despues de entrenar: ", sklearn.metrics.mean_squared_error(DataSet[1], red_neuronal(DataSet[0], theta)))

pylab.plot(rmse)
pylab.show()
