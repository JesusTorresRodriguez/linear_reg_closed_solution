import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# print(housing.DESCR)

x = housing.data[:,0]
y = housing.target

plt.scatter(x,y, alpha=0.3)

# Fórmula para minimizar el error cuadrático medio (ECM): = X(X^⊺*X)^−1X^⊺*y
# Recordar: Nos indica, en promedio, qué tan lejos están las predicciones de un modelo de los valores reales.
# por lo tanto estre más pequeño sea ese valor mas preciso será
#Dado que el producto cartesiano discriminará en el resultado el producto de las x con sigo mismo, se debe de agregar una columna de 1s
#esto para que el resultado contemple las variables independientes
x = np.array([np.ones(x.size), x]).T
# x.T significa la transpuesta de x lo mismo a x.transpose
# el simbolo @ significa multiplicación matricial
# recordar que la multiplicación escalar no es posible
# como esta ^-1 hay que sacarle la inversa con np.linalg.inv
B = np.linalg.inv(x.T @ x) @ x.T @ y
#validando los resultados .45085577, es el punto del eje Y cuando x = 0
# por otro lado .41793849 corresponde al valor de la pendiente
print("Value B:",B)

#Graficamos pendiente
plt.scatter(housing.data[:,0],y, alpha=0.3)
plt.plot([0,14],[B[0], B[1] * 14], color='red')
plt.show()