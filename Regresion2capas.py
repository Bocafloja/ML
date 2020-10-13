import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import sklearn.metrics
import pylab

# Generando los datos
ejemplos = 1000
caracteristicas = 100
D = (npr.randn(ejemplos, caracteristicas), npr.randn(ejemplos))

# Especificando la red
units_capa1 = 10
units_capa2 = 1
w1 =npr.rand(caracteristicas, units_capa1)
b1 = npr.rand(units_capa1)
w2 = npr.rand(units_capa1, units_capa2)
b2 = 0.0
theta = (w1, b1, w2, b2)

# Función de costo
def costo_cuadratico(y, y_barra):
    return np.dot((y - y_barra), (y - y_barra))

# Capa de salida
def entropia_cruzada_binaria(y, y_barra):
    return np.sum(-((y * np.log(y_barra)) + ((1-y) * np.log(1 - y_barra))))

# Armando la red
def red_neuronal(x, theta):
    w1, b1, w2, b2 = theta
    return np.tanh(np.dot((np.tanh(np.dot(x, w1) + b1)), w2) + b2)

# Armando la función objetivo a ser optimizada
def objetivo(theta, idx):
    return costo_cuadratico(D[1][idx], red_neuronal(D[0][idx], theta))

# Actualizar
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
epochs = 10
print("RMSE antes de entrenar: ", sklearn.metrics.mean_squared_error(D[1],red_neuronal(D[0],theta)))
rmse = []
for i in range(0, epochs):
    for j in range(0, ejemplos):
        delta = grad_objetivo(theta, j)
        theta = actualizar_theta(theta, delta, 0.01)
        rmse.append(sklearn.metrics.mean_squared_error(D[1], red_neuronal(D[0], theta)))

print("RMSE despues de entrenar: ", sklearn.metrics.mean_squared_error(D[1], red_neuronal(D[0], theta)))
pylab.plot(rmse)
pylab.show()