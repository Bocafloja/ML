import pylab
import numpy

x = numpy.linspace(-1, 1, 100)
signal = 2 + x + 2*x*x
ruido = numpy.random.normal(0, 0.1, 100)
y = signal + ruido
x_train = x[0:80]
y_train = y[0:80]

train_rmse = []
rmse_prueba = []
grado = 80
lambda_valores_reg = numpy.linspace(0.01, 0.99, 100)

for lambda_reg in lambda_valores_reg:
    X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, grado)])
    modelo = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(), X_train) +
                                                  lambda_reg*numpy.identity(grado)), X_train.transpose()), y_train)
    prediccion = numpy.dot(modelo, [numpy.power(x, i) for i in range(0, grado)])
    train_rmse.append(numpy.sqrt(numpy.sum(numpy.dot(y[0:80]-prediccion[0:80],
                                                     y_train-prediccion[0:80]))))
    rmse_prueba.append(numpy.sqrt(numpy.sum(numpy.dot(y[80:]-prediccion[80:],
                                                      y[80:]-prediccion[80:]))))
pylab.figure()
pylab.plot(lambda_valores_reg, train_rmse)
pylab.plot(lambda_valores_reg, rmse_prueba)
pylab.xlabel(r'$\Lambda$')
pylab.ylabel('RMSE')
pylab.legend(['Entrenamiento', 'Prueba'], loc=2)
pylab.show()
