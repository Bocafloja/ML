# Generando base de datos de prueba

import pylab
import numpy

x = numpy.linspace(-1, 1, 100)
signal = 2 + x + 2*x*x
ruido = numpy.random.normal(0, 0.1, 100)  # 100 datos con media en 0 y desviación de 0.1
y = signal + ruido
pylab.plot(signal, 'b')
pylab.plot(y, 'g')
pylab.plot(ruido, 'r')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.legend(['Sin ruido', 'Con ruido', 'Ruido'], loc=2)
x_train = x[0:80]
y_train = y[0:80]

# Modelo con grado 1
pylab.figure()
grado = 2
X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, grado)])
modelo = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(),X_train)), X_train.transpose()), y_train)
pylab.plot(x, y,'g')
pylab.title('Grado 1')
pylab.xlabel('x')
pylab.ylabel('y')
prediccion = numpy.dot(modelo, [numpy.power(x, i) for i in range(0, grado)])
pylab.plot(x, prediccion, 'r')
pylab.legend(['Señal', 'Predicción'], loc=2)
train_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80]- prediccion[0:80], y[0:80]-prediccion[0:80])))
prueba_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[80:]-prediccion[80:], y[80:]-prediccion[80:])))
print('Train RMSE (Grado = 1):', train_rmse1)
print('RMSE Prueba (Grado = 1):', prueba_rmse1)

# Modelo grado 2
pylab.figure()
grado = 3
X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, grado)])
modelo = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(),X_train)), X_train.transpose()), y_train)
pylab.plot(x, y,'g')
pylab.title('Grado 2')
pylab.xlabel('x')
pylab.ylabel('y')
prediccion = numpy.dot(modelo, [numpy.power(x, i) for i in range(0, grado)])
pylab.plot(x, prediccion, 'r')
pylab.legend(['Señal', 'Predicción'], loc=2)
train_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80]- prediccion[0:80], y[0:80]-prediccion[0:80])))
prueba_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[80:]-prediccion[80:], y[80:]-prediccion[80:])))
print('Train RMSE (Grado = 2):', train_rmse1)
print('RMSE Prueba (Grado = 2):', prueba_rmse1)


# Modelo grado 8
pylab.figure()
grado = 9
X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, grado)])
modelo = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(),X_train)), X_train.transpose()), y_train)
pylab.plot(x, y, 'g')
pylab.title('Grado 8')
pylab.xlabel('x')
pylab.ylabel('y')
prediccion = numpy.dot(modelo, [numpy.power(x, i) for i in range(0, grado)])
pylab.plot(x, prediccion, 'r')
pylab.legend(['Señal', 'Predicción'], loc=2)
train_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80]- prediccion[0:80], y[0:80]-prediccion[0:80])))
prueba_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[80:]-prediccion[80:], y[80:]-prediccion[80:])))
print('Train RMSE (Grado = 8):', train_rmse1)
print('RMSE Prueba (Grado = 8):', prueba_rmse1)
pylab.show()