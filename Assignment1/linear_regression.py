import numpy as np

class LinearRegression():
    
    def __init__(self, lr, epochs):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.lr = lr
        self.epochs = epochs
        self.theta0 = 0
        self.theta1 = 0
        self.residuals = []
        pass
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        for _ in range(self.epochs):
            # regresion formula
            y_hat = self.theta0 + self.theta1*X

            error = y_hat - y
            self.residuals.append(error)

            grad_theta0 = error.mean()
            grad_theta1 = (error*X).mean()

            self.theta0 -= self.lr*grad_theta0
            self.theta1 -= self.lr*grad_theta1

            
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        return self.theta0 + self.theta1*X





