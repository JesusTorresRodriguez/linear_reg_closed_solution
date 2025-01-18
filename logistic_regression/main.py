import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.weights = None
        self.loss_history = []
        self.std_loss_history = []

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    def log_loss(self,y,prob):
        """
        pred_probs must be in the shape of (number_of_samples,number_of_categories)
        """
        # Clip probabilities to avoid log(0) or log(1) instability
        prob = np.clip(prob, 1e-15, 1 - 1e-15)
        # Compute log loss
        log_loss = (y * np.log(prob)) + ((1-y) * np.log(1-prob))
        return -1*log_loss
    def fit(self, X, y, epochs=1000, lr=0.01):
        # Add bias term to X
        ones = np.ones(X.shape[0])
        X = np.insert(X, 0, ones, axis=1)
        
        # Initialize weights (including bias)
        n_samples, n_features = X.shape
        self.weights = np.ones(n_features)
        self.weights = self.weights/2

        # Gradient Descent
        for _ in range(epochs):
            linear_projection = np.dot(X, self.weights)
            y_predicted = self.sigmoid(linear_projection)#transforma los datos continuos a datos entre 0 y 1
            loss =  self.log_loss(y,y_predicted)
            avg_loss = np.sum(loss)/loss.size
            std_loss = np.std(loss)
            self.loss_history.append(avg_loss)
            self.std_loss_history.append(std_loss)
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            # Update weights
            self.weights -= lr * dw

    def predict(self, X):
        # Add bias term to X
        ones = np.ones(X.shape[0])
        X = np.insert(X, 0, ones, axis=1)
        return self.sigmoid(np.dot(X, self.weights))
class MyLabelEncoder:
    def __init__(self):
        self.class_to_num = {}
        self.num_to_class = {}

    def fit(self, y):
        #Fit label encoder to y.
        classes = np.unique(y)
        for i, cls in enumerate(classes):
            self.class_to_num[cls] = i
            self.num_to_class[i] = cls

    def transform(self, y):
        #Transform labels to normalized encoding.
        return np.array([self.class_to_num[cls] for cls in y])

    def inverse_transform(self, y_encoded):
        #Transform labels back to original encoding.
        return np.array([self.num_to_class[num] for num in y_encoded])
def get_data(standarize=False):
    data = pd.read_csv('dataset.csv', delimiter=';', header=0)
    data_one_hot = pd.get_dummies(data, columns=['Competition', 'PlayerType','Movement'])
    le = MyLabelEncoder()
    le.fit(data_one_hot['ShotType'])
    # Fit the encoder to the labels of the 'TargetColumn'
    data_one_hot['ShotType_Encoded'] = le.transform(data_one_hot['ShotType'])
    feature_cols = ['Angle', 'Distance','Competition_EURO', 'Competition_NBA',
                        'Competition_SLO1','Competition_U14','Competition_U16',
                        'Movement_dribble or cut','Movement_drive','Movement_no',
                        'PlayerType_C','PlayerType_F','PlayerType_G',
                        'Transition','TwoLegged']
    target_col = ['ShotType_Encoded']
    if standarize:
        cols_to_scale = ['Angle','Distance']
        features = data_one_hot[feature_cols].copy()

        features[cols_to_scale] = (data_one_hot[cols_to_scale] - data_one_hot[cols_to_scale].mean()) / data_one_hot[cols_to_scale].std()
        #features_scaled_df[feature_cols] = (data_one_hot[feature_cols] - data_one_hot[feature_cols].mean()) / data_one_hot[feature_cols].std()
    else:
        features = data_one_hot[feature_cols].copy()
    target_df = data_one_hot[target_col].copy()
    X = features.values.astype(float)
    y = np.squeeze(target_df.values)
    return X,y,feature_cols #returning X, y and feature col names
# Example usage
if __name__ == "__main__":
    # Generate some data
    dataset_housingr = pd.read_csv('housing3.csv')
    features = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    target = ['Class'] #target, true value 
    
    X =  dataset_housingr[features].values
    X = StandardScaler().fit_transform(X)
    le = MyLabelEncoder()
    le.fit(dataset_housingr['Class'])
    dataset_housingr['target_Encoded'] = le.transform(dataset_housingr['Class'])
    y = np.squeeze(dataset_housingr['target_Encoded'].values)
    # Train model
    model = LogisticRegression()
    model.fit(X, y, epochs=1000, lr=0.01)

    # Make predictions
    predictions = model.predict(X)
    # Get the index and value of the minimum loss
    min_loss_index = np.argmin(model.loss_history)
    min_loss_value = model.loss_history[min_loss_index]

    # Get the index and value of the minimum standard deviation of loss
    min_std_loss_index = np.argmin(model.std_loss_history)
    min_std_loss_value = model.std_loss_history[min_std_loss_index]

    # Print the results
    print(f"Minimum loss: {min_loss_value} at epoch {min_loss_index}")
    print(f"Minimum STD loss at minimum loss: {model.std_loss_history[min_loss_index]} at epoch {min_loss_index}")
    print(f"Minimum STD loss: {min_std_loss_value} at epoch {min_std_loss_index}")
    # Evaluate model
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