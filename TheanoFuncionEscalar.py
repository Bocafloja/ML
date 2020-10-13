import theano.tensor as T
from theano import function

a = T.dscalar('a')
b = T.dscalar('b')
c = T.dscalar('c')
d = T.dscalar('d')
e = T.dscalar('e')

f = ((a - b + c) * d)/e

g = function([a, b, c, d, e], f)

print('Valor esperado: ((1 - 2 + 3) * 4)/5.0 = ', ((1 - 2 + 3) * 4)/5.0)
print('Usando Theano: ((1 - 2 + 3) * 4)/5.0 = ', g(1, 2, 3, 4, 5))
