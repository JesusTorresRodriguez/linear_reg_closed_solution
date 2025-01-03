import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.loss_history = []
        self.std_loss_history = []        

    def fit(self, x, y, epochs=1000, lr=0.01):
        # Agrega el bias a X
        # agrega fila de 1s a la matriz X
        ones = np.ones(x.shape[0])
        x = np.insert(x, 0, ones, axis=1)
       
        
        # Initialize weights (including bias)
        n_samples, n_features = x.shape
        self.weights = np.ones(n_features)
        self.weights = self.weights / 2

        # Gradiente decendiente
        # El gradiente decendiente es un algoritmo de optimización que se utiliza para minimizar una función
        # ajustando iterativamente los parámetros de un modelo.
        # En el caso de la regresión lineal, el objetivo es minimizar la función de pérdida
        # que es el error cuadrático medio (MSE) entre las predicciones y los valores reales.
        for _ in range(epochs):
            #cálculo del producto punto
            y_predicted = np.dot(x, self.weights)
            # loss =  (y-y_predicted)**2
            # avg_loss = np.sum(loss)/loss.size
            # std_loss = np.std(loss)
            # self.loss_history.append(avg_loss)
            # self.std_loss_history.append(std_loss)
            # Calculo del gradiente
            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))

            # Actualización de los pesos
            self.weights -= lr * dw

    def predict(self, x):
        # Add bias term to X
        ones = np.ones(x.shape[0])
        x = np.insert(x, 0, ones, axis=1)
        # se calcula el producto punto contra los pesos
        return np.dot(x, self.weights)

    def score(self, x, y):
        y_pred = self.predict(x)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

if __name__ == "__main__":
    # dataset_housingr = pd.read_csv('./reg_linear_gradient_desc/housing2r.csv')
    # features = ['RM', 'AGE', 'DIS', 'RAD', 'TAX']
    # target = ['y'] #target, true value
    # x, y = dataset_housingr[features].values, np.squeeze(dataset_housingr[target].values) 
    # # Normaliza la data de entrada
    # x = StandardScaler().fit_transform(x)
    x = np.array([0, 8, 15, 22, 38, 40])
    x = x.reshape((6, 1))
    y= np.array([ 32, 46, 59, 62, 72, 100])

    # Entrena el modelo
    model = LinearRegression()
    model.fit(x, y, epochs=1000, lr=0.01)

    # Make predictions
    predictions = model.predict(x)
    # Plot the data and the regression line
    plt.figure()
    plt.scatter(x[:, 0], y, color='blue', alpha=0.5, label='Data points')
    plt.plot(x[:, 0], predictions, color='red', label='Regression line')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.show()
    # loss =  (y-predictions)**2
    # avg_loss = np.sum(loss)/loss.size
    # std_loss = np.std(loss)
    # Evaluate model
    r2_score = model.score(x, y)
    print(f"R^2 Score: {r2_score:.4f}")
    # Plot loss
    plt.figure()
    plt.plot(model.loss_history, label='Average Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(model.std_loss_history, label='Standard Deviation of Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Standard Deviation')
    plt.title('Loss Standard Deviation over Epochs')
    plt.legend()
    # plt.show()