import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.weights = None
        self.loss_history = []
        self.std_loss_history = []

    def fit(self, X, y, epochs=1000, lr=0.01):
        # Agrega el bias a X
        # agrega fila de 1s a la matriz X
        ones = np.ones(X.shape[0])
        X = np.insert(X, 0, ones, axis=1)
        
        # Initialize weights (including bias)
        n_samples, n_features = X.shape
        self.weights = np.ones(n_features)
        self.weights = self.weights/2

        # Gradiente decendiente
        # El gradiente decendiente es un algoritmo de optimización que se utiliza para minimizar una función
        # ajustando iterativamente los parámetros de un modelo.
        # En el caso de la regresión lineal, el objetivo es minimizar la función de pérdida
        # que es el error cuadrático medio (MSE) entre las predicciones y los valores reales.
        for _ in range(epochs):
            y_predicted = np.dot(X, self.weights)
            loss =  (y-y_predicted)**2
            avg_loss = np.sum(loss)/loss.size
            std_loss = np.std(loss)
            self.loss_history.append(avg_loss)
            self.std_loss_history.append(std_loss)
            # Calculo del gradiente
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            # Actualización de los pesos
            self.weights -= lr * dw

    def predict(self, X):
        # Add bias term to X
        ones = np.ones(X.shape[0])
        X = np.insert(X, 0, ones, axis=1)
        return np.dot(X, self.weights)

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

# Example usage
if __name__ == "__main__":
    # Generate some data
    '''np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100)'''
    dataset_housingr = pd.read_csv('housing2r.csv')
    features = ['RM', 'AGE', 'DIS', 'RAD', 'TAX']
    target = ['y'] #target, true value
    X, y = dataset_housingr[features].values, np.squeeze(dataset_housingr[target].values) 
    X = StandardScaler().fit_transform(X)
    # Train model
    model = LinearRegression()
    model.fit(X, y, epochs=1000, lr=0.01)

    # Make predictions
    predictions = model.predict(X)
    loss =  (y-predictions)**2
    avg_loss = np.sum(loss)/loss.size
    std_loss = np.std(loss)
    # Evaluate model
    r2_score = model.score(X, y)
    print(f"R^2 Score: {r2_score:.4f}")
    # Plot loss
    plt.figure()
    plt.plot(model.loss_history, label='Average Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('avg_loss.png')

    plt.figure()
    plt.plot(model.std_loss_history, label='Standard Deviation of Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Standard Deviation')
    plt.title('Loss Standard Deviation over Epochs')
    plt.legend()
    plt.savefig('std_loss.png')