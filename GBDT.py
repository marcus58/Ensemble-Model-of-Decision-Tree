import copy
import numpy as np


class GBDT:
    '''Adaboost Classifier.

    Note that this class only support binary classification.
    '''

    def __init__(self,
                 base_learner,
                 n_estimator,
                 learning_rate,
                 seed=2020):
        '''Initialize the classifier.

        Args:
            base_learner: the base_learner should provide the .fit() and .predict() interface.
            n_estimator (int): The number of base learners in RandomForest.
            seed (int): random seed
        '''
        np.random.seed(seed)
        self.base_learner = base_learner
        self.n_estimator = n_estimator
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.n_estimator)]
        self.learning_rate=learning_rate

    def fit(self, X, y):
        """Build the Adaboost according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        _y=np.array(y)
        pos=np.sum(_y)
        neg=len(_y)-pos
        y_pred = np.ones(_y.shape)*np.log(pos/neg)
        self.f_0=y_pred
        for i in range(self.n_estimator):
            residual= _y-1/(1+np.exp(-y_pred)) # negative gradient
            self._estimators[i].fit(X, residual, _y)
            y_pred+=np.multiply(self.learning_rate, self._estimators[i].predict(X))
        # end answer
        return self

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        y_pred = np.zeros(len(X))
        # YOUR CODE HERE
        # begin answer
        for i in range(self.n_estimator):
            y_pred += np.multiply(self.learning_rate, self._estimators[i].predict(X))           
        
        # Turn into probability distribution
        y_pred=1/(1+np.exp(-y_pred))
        y_pred=(y_pred>=0.5).astype('int')

        # end answer
        return y_pred
