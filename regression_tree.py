import numpy as np

class RegressionTree:
    '''Decision Tree Regression.
    '''

    def __init__(self,
                 max_depth,
                 min_samples_leaf,
                 sample_feature=False):
        '''Initialize the classifier.

        Args:
            max_depth (int): the max depth for the decision tree. This parameter is
                a trade-off between underfitting and overfitting.
            min_samples_leaf (int): the minimal samples in a leaf. This parameter is a trade-off
                between underfitting and overfitting.
            sample_feature (bool): whether to sample features for each splitting. Note that for random forest,
                we would randomly select a subset of features for learning. Here we select sqrt(p) features.
                For single decision tree, we do not sample features.
        '''
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature

    def fit(self, X, y, label, sample_weights=None):
        """Build the decision tree according to the training data.

        Args:
            X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
            y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] <= C. Here C = 1.
            sample_weights: weights for each samples, of shape (N,).
        """
        if sample_weights is None:
            # if the sample weights is not provided, then by default all
            # the samples have unit weights.
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, label, feature_names, depth=1, sample_weights=sample_weights)
        return self

    @staticmethod
    def _impurity(y, y1, y2, sample_weights=None):
        """Calculate the gini impurity for labels.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the gini impurity for y.
        """
        # YOUR CODE HERE
        # begin answer
        weight_1=len(y1)
        weight_2=len(y2)
        meal_1=np.sum(y1)/float(weight_1)
        meal_2=np.sum(y2)/float(weight_2)
        diff=meal_1-meal_2
        sum_var=weight_1*weight_2*diff*diff/(weight_1+weight_2)
        # end answer
        return sum_var

    def _split_dataset(self, X, y, label, index, value, sample_weights=None):
        """Return the split of data whose index-th feature equals value.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for splitting.
            value: the value of the index-th feature for splitting.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (np.array): the subset of X whose index-th feature equals value.
            (np.array): the subset of y whose index-th feature equals value.
            (np.array): the subset of sample weights whose index-th feature equals value.
        """
        # YOUR CODE HERE
        # Hint: Do not forget to remove the index-th feature from X.
        # begin answer
        ret1=[]
        ret2=[]
        featVec=X[:,index]
        X=X[:,[i for i in range(X.shape[1]) if i!=index ]]
        for i in range(len(featVec)):
            if featVec[i]>=value:
                ret1.append(i)
            else:
                ret2.append(i)
        sub1_X = X[ret1,:]
        sub1_y = y[ret1]
        label_1=label[ret1]
        sub1_sample_weights=sample_weights[ret1]
        sub2_X = X[ret2,:]
        sub2_y = y[ret2]
        label_2=label[ret2]
        sub2_sample_weights=sample_weights[ret2]
        # end answer
        return sub1_X, sub1_y, label_1, sub1_sample_weights, sub2_X, sub2_y, label_2, sub2_sample_weights

    def _choose_best_feature(self, X, y, label, sample_weights=None):
        """Choose the best feature to split according to criterion.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the index for the best feature
        """
        best_feature_idx = 0
        # YOUR CODE HERE
        # Note that you need to implement the sampling feature part here for random forest!
        # Hint: You may find `np.random.choice` is useful for sampling.
        # begin answer
        n_features = X.shape[1]
        if self.sample_feature:
            max_features=max(1, min(n_features, int(np.round(np.sqrt(n_features)))))
            new_features=np.random.choice(n_features, max_features, replace=False)
            new_X=X[:, new_features]
        else:
            new_X=X
        n_new_features=new_X.shape[1]
        #new_features=np.random.choice(n_features, n_features, replace=False)
        #old_cost=self.entropy(y, sample_weights)
        #use C4.5 algorirhm
        best_impurity=None
        best_feature_idx=0
        best_feature_val=X[0, 0]
        for i in range(n_new_features):
            unique_vals=np.unique(X[:,i])
            for value in unique_vals:
                sub1_X, sub1_y, label1, sub1_sample_weights, sub2_X, sub2_y, label2, sub2_sample_weights=self._split_dataset(X, y, label, i, value, sample_weights)
                if len(sub1_y)>0 and len(sub2_y)>0:
                    new_impurity=self._impurity(y, sub1_y, sub2_y)
                    if best_impurity is None or new_impurity > best_impurity:
                        best_impurity=new_impurity
                        best_feature_idx=i
                        best_feature_val=value               
        # end answer
        return best_feature_idx, best_feature_val

    @staticmethod
    def _leaf_calculation(y, label, sample_weights=None):
        """Return the label which appears the most in y.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the majority label
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        # YOUR CODE HERE
        # begin answer
        numerator=np.sum(y)
        denominator=np.sum((label-y)*(1-label+y))
        if numerator == 0 or abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator/denominator

    def _build_tree(self, X, y, label, feature_names, depth, sample_weights=None):
        """Build the decision tree according to the data.

        Args:
            X: (np.array) training features, of shape (N, D).
            y: (np.array) vector of training labels, of shape (N,).
            feature_names (list): record the name of features in X in the original dataset.
            depth (int): current depth for this node.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (dict): a dict denoting the decision tree. 
            Example:
                The first best feature name is 'title', and it has 5 different values: 0,1,2,3,4. For 'title' == 4, the next best feature name is 'pclass', we continue split the remain data. If it comes to the leaf, we use the majority_label by calling majority_vote.
                mytree = {
                    'titile': {
                        0: subtree0,
                        1: subtree1,
                        2: subtree2,
                        3: subtree3,
                        4: {
                            'pclass': {
                                1: majority_vote([1, 1, 1, 1]) # which is 1, majority_label
                                2: majority_vote([1, 0, 1, 1]) # which is 1
                                3: majority_vote([0, 0, 0]) # which is 0
                            }
                        }
                    }
                }
        """
        mytree = dict()
        # YOUR CODE HERE
        # TODO: Use `_choose_best_feature` to find the best feature to split the X. Then use `_split_dataset` to
        # get subtrees.
        # Hint: You may find `np.unique` is useful.
        # begin answer
        #1. no feature 2. all lables are the same 3. depth exceed 4. X is too small
        if len(feature_names)==0 or len(np.unique(y))==1 or depth >= self.max_depth or len(X) <= self.min_samples_leaf: 
            return self._leaf_calculation(y, label, sample_weights)
        best_feature_idx, best_feature_val=self._choose_best_feature(X, y, label, sample_weights)
        best_feature_name = feature_names[best_feature_idx]
        feature_names=feature_names[:]
        feature_names.remove(best_feature_name)
        mytree={best_feature_name:{}}
        sub1_X, sub1_y, label1, sub1_sample_weights, sub2_X, sub2_y, label2, sub2_sample_weights = self._split_dataset(X, y, label, best_feature_idx, best_feature_val, sample_weights)
        mytree[best_feature_name][(best_feature_val, True)]=self._build_tree(sub1_X, sub1_y, label1, feature_names, depth+1, sub1_sample_weights)
        mytree[best_feature_name][(best_feature_val, False)]=self._build_tree(sub2_X, sub2_y, label2, feature_names, depth+1, sub2_sample_weights)
        # end answer
        return mytree

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: (pd.Dataframe) testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        def _classify(tree, x):
            """Classify a single sample with the fitted decision tree.

            Args:
                x: ((pd.Dataframe) a single sample features, of shape (D,).

            Returns:
                (int): predicted testing sample label.
            """
            # YOUR CODE HERE
            # begin answer
            feature_name=list(tree.keys())[0] #first element
            secondDict=tree[feature_name]            
            key=x.loc[feature_name] #extract value from x
            for key_val in secondDict:
                feature_val=key_val[0]
            valueOfKey=secondDict[(feature_val, key>=feature_val)]
            if isinstance(valueOfKey,dict):
                label=_classify(valueOfKey,x)
            else:
                label=valueOfKey
            return label
            # end answer

        # YOUR CODE HERE
        # begin answer
        if len(X.shape)==1:
            return _classify(self._tree,X)
        else:
            results=[]
            for i in range(X.shape[0]):
                results.append(_classify(self._tree, X.iloc[i, :]))
            return np.array(results)
        # end answer

    def show(self):
        """Plot the tree using matplotlib
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)

