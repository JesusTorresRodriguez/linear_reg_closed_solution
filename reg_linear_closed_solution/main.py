import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
class LinearRegression:
    def __init__(self):
        self.B = None
        self.x = None
        self.y = None

    def fit(self,x,y):
        self.x = x
        self.y = y
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
        self.B = np.linalg.inv(x.T @ x) @ x.T @ y
        #validando los resultados B[0], es el punto del eje Y cuando x = 0
        # por otro lado B[1] corresponde al valor de la pendiente
        print("Value B:",self.B)
    
    def plot(self):
        #Graficamos pendiente
        # plt.scatter(housing.data[:,0],y, alpha=0.3)
        plt.scatter(self.x,self.y, alpha=0.3)
        # plt.plot([0,14],[B[0], B[1] * 14], color='red')
        plt.plot([min(self.x),max(self.x)],[self.B[0] + self.B[1] * min(self.x), self.B[0] + self.B[1] * max(self.x)], color='red')

    def predict(self,x_pred):
        # predicción
        y_pred = self.B[0] + self.B[1] * x_pred
        print("Predicción:", y_pred)
        # Agrega el punto predicho al gráfico
        plt.scatter(x_pred, y_pred, color='blue', zorder=5)
        plt.show()




if __name__ == "__main__":
    # x = housing.data[:,0]
    # y = housing.target
    x= np.array([ 0, 8, 15, 22, 38, 40])
    y= np.array([ 32, 46, 59, 62, 72, 100])

    model = LinearRegression()
    model.fit(x,y)
    model.plot()
    model.predict(10)
  

   