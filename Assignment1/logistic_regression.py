import numpy as np

class LogisticRegression():

    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.b = 0
        self.w0 = 0
        self.w1 = 0
        self.residuals = []
        pass

    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
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

    

    def predict(self, X):
        return self._sigmoid(self.b + self.w0*X[:, 0] + self.w1*X[:, 1])

    def accuracy(self, X, y, threshold=0.5):
        y_hat = self.predict(X)
        y_pred = (y_hat >= threshold).astype(int)
        return np.mean(y == y_pred)
    