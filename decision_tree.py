import numpy as np

class DecisionTree:
    '''Decision Tree Classifier.

    Note that this class only supports binary classification.
    '''

    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_leaf,
                 sample_feature=False):
        '''Initialize the classifier.

        Args:
            criterion (str): the criterion used to select features and split nodes.
            max_depth (int): the max depth for the decision tree. This parameter is
                a trade-off between underfitting and overfitting.
            min_samples_leaf (int): the minimal samples in a leaf. This parameter is a trade-off
                between underfitting and overfitting.
            sample_feature (bool): whether to sample features for each splitting. Note that for random forest,
                we would randomly select a subset of features for learning. Here we select sqrt(p) features.
                For single decision tree, we do not sample features.
        '''
        if criterion == 'infogain_ratio':
            self.criterion = self._information_gain_ratio
        elif criterion == 'entropy':
            self.criterion = self._information_gain
        elif criterion == 'gini':
            self.criterion = self._gini_purification
        else:
            raise Exception('Criterion should be infogain_ratio or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature

    def fit(self, X, y, sample_weights=None):
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
        self._tree = self._build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)
        return self

    @staticmethod
    def entropy(y, sample_weights=None):
        """Calculate the entropy for label.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the entropy for y.
        """
        entropy = 0.0
        # begin answer
        num=y.shape[0]#number of labels
        labelCounts={}#caculate different labels in y，and store in labelCounts
        #for label in y:
        #    if label not in labelCounts.keys():
        #        labelCounts[label]=0
        #    labelCounts[label]+=1
        for i in range(num):
            if y[i] not in labelCounts.keys():
                labelCounts[y[i]]=0
            labelCounts[y[i]]+=sample_weights[i]
        for key in labelCounts:
            prob=float(labelCounts[key])/float(np.sum(sample_weights))
            entropy-=prob*np.log2(prob)
        # end answer
        return entropy

    def _information_gain(self, X, y, index, sample_weights=None):
        """Calculate the information gain given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain calculated.
        """
        info_gain = 0
        # YOUR CODE HERE
        # begin answer
        old_cost=self.entropy(y, sample_weights)
        unique_vals=np.unique(X[:,index])
        new_cost=0.0
        #split the values of i-th feature and calculate the cost 
        for value in unique_vals:
            sub_X,sub_y, sub_sample_weights=self._split_dataset(X, y, index, value, sample_weights)
            #prob=len(sub_y)/float(len(y))
            prob=np.sum(sub_sample_weights)/float(np.sum(sample_weights))
            new_cost+=prob*self.entropy(sub_y, sub_sample_weights)
        #if split_information==0, then all values of i-th feature are the same, so i-th feature cannot be best feature
        info_gain=old_cost-new_cost #info gain
        # end answer
        return info_gain

    def _information_gain_ratio(self, X, y, index, sample_weights=None):
        """Calculate the information gain ratio given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain ratio calculated.
        """
        info_gain_ratio = 0
        split_information = 0.0
        # YOUR CODE HERE
        # begin answer
        old_cost=self.entropy(y, sample_weights)
        unique_vals=np.unique(X[:,index])
        new_cost=0.0
        split_information=0.0
        #split the values of i-th feature and calculate the cost 
        for value in unique_vals:
            sub_X,sub_y, sub_sample_weights=self._split_dataset(X, y, index, value, sample_weights)
            #prob=len(sub_y)/float(len(y))
            prob=np.sum(sub_sample_weights)/float(np.sum(sample_weights))
            new_cost+=prob*self.entropy(sub_y, sub_sample_weights)
            split_information-=prob*np.log2(prob)
        #if split_information==0, then all values of i-th feature are the same, so i-th feature cannot be best feature
        if split_information==0.0:
            pass
        else:
            info_gain=old_cost-new_cost #info gain
            info_gain_ratio=info_gain/split_information #info gain ratio
        # end answer
        return info_gain_ratio

    @staticmethod
    def gini_impurity(y, sample_weights=None):
        """Calculate the gini impurity for labels.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the gini impurity for y.
        """
        gini = 1
        # YOUR CODE HERE
        # begin answer
        num=y.shape[0]#number of labels
        labelCounts={}#caculate different labels in y，and store in labelCounts
        #for label in y:
        #    if label not in labelCounts.keys():
        #        labelCounts[label]=0
        #    labelCounts[label]+=1
        for i in range(num):
            if y[i] not in labelCounts.keys():
                labelCounts[y[i]]=0
            labelCounts[y[i]]+=sample_weights[i]
        for key in labelCounts:
            prob=float(labelCounts[key])/float(np.sum(sample_weights))
            gini -= prob ** 2
        # end answer
        return gini

    def _gini_purification(self, X, y, index, sample_weights=None):
        """Calculate the resulted gini impurity given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the resulted gini impurity after splitting by this feature.
        """
        new_impurity = 0
        # YOUR CODE HERE
        # begin answer
        old_cost=self.gini_impurity(y, sample_weights)
        unique_vals=np.unique(X[:,index])
        new_cost=0.0
        #split the values of i-th feature and calculate the cost 
        for value in unique_vals:
            sub_X,sub_y, sub_sample_weights=self._split_dataset(X, y, index, value, sample_weights)
            #prob=len(sub_y)/float(len(y))
            prob=np.sum(sub_sample_weights)/float(np.sum(sample_weights))
            new_cost+=prob*self.gini_impurity(sub_y, sub_sample_weights)
        #if split_information==0, then all values of i-th feature are the same, so i-th feature cannot be best feature
        new_impurity=old_cost-new_cost #info gain
        # end answer
        return new_impurity

    def _split_dataset(self, X, y, index, value, sample_weights=None):
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
        ret=[]
        featVec=X[:,index]
        X=X[:,[i for i in range(X.shape[1]) if i!=index ]]
        for i in range(len(featVec)):
            if featVec[i]==value:
                ret.append(i)
        sub_X = X[ret,:]
        sub_y = y[ret]
        sub_sample_weights=sample_weights[ret]
        # end answer
        return sub_X, sub_y, sub_sample_weights

    def _choose_best_feature(self, X, y, sample_weights=None):
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
        best_gain_cost=0.0
        for i in range(n_new_features):
            info_gain_cost=self.criterion(new_X,y,i,sample_weights)           
            if info_gain_cost > best_gain_cost:
                best_gain_cost=info_gain_cost
                best_feature_idx=i                
        # end answer
        return best_feature_idx

    @staticmethod
    def majority_vote(y, sample_weights=None):
        """Return the label which appears the most in y.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the majority label
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        majority_label = y[0]
        # YOUR CODE HERE
        # begin answer
        dict_num={}
        for i in range(y.shape[0]):
            if y[i] not in dict_num.keys():
                dict_num[y[i]]=sample_weights[i]
            else:
                dict_num[y[i]] += sample_weights[i]
        majority_label=max(dict_num, key=dict_num.get)
        # end answer
        return majority_label

    def _build_tree(self, X, y, feature_names, depth, sample_weights=None):
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
            return self.majority_vote(y, sample_weights)
        best_feature_idx=self._choose_best_feature(X, y, sample_weights)
        best_feature_name = feature_names[best_feature_idx]
        feature_names=feature_names[:]
        feature_names.remove(best_feature_name)
        mytree={best_feature_name:{}}
        unique_vals=np.unique(X[:, best_feature_idx])
        for value in unique_vals:
            sub_X, sub_y, sub_sample_weights = self._split_dataset(X, y, best_feature_idx, value, sample_weights)
            mytree[best_feature_name][value]=self._build_tree(sub_X, sub_y, feature_names, depth+1, sub_sample_weights)
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
            key=x.loc[feature_name]
            if key not in secondDict:
                key=np.random.choice(list(secondDict.keys()))
            valueOfKey=secondDict[key]
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