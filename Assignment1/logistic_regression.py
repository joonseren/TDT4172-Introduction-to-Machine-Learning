import numpy as np

class LogisticRegression():

    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.wheights= []
        self.b = 0
        self.w0 = 0
        self.w1 = 0
        self.residuals = []
        pass

    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def fit1(self, X, y):
        """
        Estimates parameters for the classifier only two wheights
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """

        for _ in range(self.epochs):
            # lineær modell og sigmoid
            z = self.b + self.w0*X[:, 0] + self.w1*X[:, 1]
            y_hat = self._sigmoid(z)
            
            # oppdaterer residualer
            self.residuals.append(y_hat - y)

            # gradienter av log loss tapsfunksjon
            db = np.mean(y_hat - y)
            dw0 = np.mean((y_hat - y)*X[:, 0])
            dw1 = np.mean((y_hat - y)*X[:, 1])

            # oppdaterer vekter og biaser
            self.b -= self.lr*db
            self.w0 -= self.lr*dw0
            self.w1 -= self.lr*dw1

    def fit2(self, X, y):
        # bruker array for vekter og biaser
        n_samples, n_features = X.shape
        self.wheights = np.zeros(n_features + 1)

        for _ in range(self.epochs):
            # lineær modell og sigmoid
            z = self.wheights[0] + np.dot(X, self.wheights[1:])
            y_hat = self._sigmoid(z)

            # gradienter
            db = np.mean(y_hat - y)
            dw = 1/n_samples * np.dot(X.T, (y_hat -y))

            # oppdatere vekter og bias
            self.wheights[0] -= self.lr*db
            self.wheights[1:] -= self.lr*dw
        


    def predict1(self, X):
        return self._sigmoid(self.b + self.w0*X[:, 0] + self.w1*X[:, 1])
    
    def predict2(self, X):
        print("X shape:", X.shape)                     # (n_samples, n_features)
        print("weights shape:", self.wheights.shape)    # (n_features+1,)
        print("w (uten bias) shape:", self.wheights[1:].shape)  # (n_features,)
        return self._sigmoid(self.wheights[0] + X @ self.wheights[1:])

    def accuracy1(self, X, y, threshold=0.5):
        y_hat = self.predict1(X)
        y_pred = (y_hat >= threshold).astype(int)
        return np.mean(y == y_pred)
    
    def accuracy2(self, X, y, threshold=0.5):
        y_hat = self.predict2(X)
        y_pred = (y_hat >= threshold).astype(int)
        return np.mean(y == y_pred)
    