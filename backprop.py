class MLP(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=.1, momentum=0, shuffle=True, hidden_layer_widths=None, deterministic=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes. (and then one more for the bias node)
            For example: input width 1, then hidden layer will be 3 nodes
        Example:
            mlp = MLP(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.epochs = 0
        self.weights = []
        self.delta_dub = []
        self.validations_error = []
        self.training_error = []
        self.deterministic = deterministic

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """

        n = len(y)
        # num cols = amount of features

        self.X = X

        if self.shuffle:
            X, y = self._shuffle_data(X, y)

        # augment pattern with 1s to add bias
        pattern = self.augment_ones(X)

        self.X = pattern

        def activation(z):
            return 1 / (1 + np.exp(-z))

        # self.hidden_layer_widths = [inputs * 2] if self.hidden_layer_widths is None else self.hidden_layer_widths
        self.weights.append(
            self.initialize_weights(len(X[0]), self.hidden_layer_widths[0])) if not initial_weights else initial_weights
        self.delta_dub.append(self.initialize_weights(len(X[0]), self.hidden_layer_widths[0]))
        # self.weights = self.initialize_5(pattern)
        # for i in range(len())

        self.weights.append(self.initialize_weights(self.hidden_layer_widths[-1], len(y[0])))
        # print(self.hidden_layer_widths[-1])
        # print(len(y[0]))
        self.delta_dub.append(self.initialize_weights(self.hidden_layer_widths[-1], len(y[0])))

        bestAcc = np.array([0])
        bestW = self.weights
        improvement = 0
        limit = 100

        # print(pattern)
        # print(self.weights)

        trainX, trainY, varX, varY = X, y, X, y

        # print(trainX)
        # print(trainY)
        # print(varX)
        # print(varY)
        stopVar = False
        while not stopVar:
            for i in range(len(trainX)):  # (len(trainX))
                inDat = trainX[i]
                target = trainY[i]
                nets = []
                # print(inDat)
                # print(target)
                for l in self.weights:
                    inDat = np.append(inDat, 1)
                    sigmoidList = []
                    nets.append(inDat)
                    for node in l:
                        result = np.sum(inDat * node)
                        sigmoidList.append(activation(result))
                    inDat = np.array(sigmoidList)

                # print(inDat)
                outputs = inDat

                d_output = np.subtract(target, outputs) * outputs * np.subtract(1, outputs)
                # print(self.weights)
                delta_vals = [d_output]

                for back in range(len(nets) - 1, -1, -1):
                    trans_weights = np.transpose(self.weights[back])
                    # print(trans_weights)
                    jth_delta = []
                    for j in range(len(trans_weights)):
                        jth_delta.append(
                            np.sum(delta_vals[-1] * trans_weights[j]) * nets[back][j] * (1 - nets[back][j]))

                    self.delta_dub[back] = self.lr * np.outer(delta_vals[-1], nets[back]) + (
                            self.momentum * self.delta_dub[back])
                    delta_vals.append(np.array(jth_delta[:-1]))
                for i in range(len(self.weights)):
                    self.weights[i] = np.add(self.weights[i], self.delta_dub[i])
            self.epochs += 1
            print(self.epochs)
            # print(self.weights)
            # print(varX)
            # print(varY)
            self.validations_error.append(self.mse(varX, varY))
            self.training_error.append(self.mse(trainX, trainY))
            stopVar = self.epochs >= self.deterministic

        self.weights = bestW
        return self

    def mse(self, x, y):
        prediction = self.predict(x, dimension=len(y[0]))
        result = prediction - np.array(y).reshape(-1, len(y[0]))
        return np.sum(np.square(result)) / y.size

    def predict(self, X, dimension=1):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        def activation(z):
            return 1 / (1 + np.exp(-z))

        predictions = []
        for i in range((len(X))):  # (len(trainX))
            inDat = X[i]
            nets = []
            for l in self.weights:
                inDat = np.append(inDat, 1)
                sigmoidList = []
                nets.append(inDat)
                for node in l:
                    result = np.sum(inDat * node)
                    sigmoidList.append(activation(result))
                inDat = np.array(sigmoidList)

            output = []
            for i in inDat:
                if i >= 0.5:
                    output.append(1)
                else:
                    output.append(0)
            predictions.append(np.array(output))
        return np.array(predictions).reshape(-1, 1)

    def initialize_weights(self, col, row):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        rows = row
        cols = col + 1
        apple = np.zeros((rows, cols), dtype=np.double)
        # n_features = shapes.shape[1]
        # self.V = np.random.uniform()
        self.initial_weights = apple
        return apple

    # def initialize_5(self, shapes):
    #     rows = self.hidden_layer_widths[0] #hidden_layer_widths = in our case is 4
    #     cols = shapes.shape[1] #3

    #     myList = []
    #     for i in range(rows):
    #       myOther = []
    #       for j in range(cols):
    #         myOther.append(0.5)
    #       myList.append(myOther)
    #     return np.array(myList)

    def augment_ones(self, arr):

        arr = np.append(arr, 1)
        return arr

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        predictions = self.predict(X)

        return sum(predictions == y) / len(y)

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        shuffled = np.random.permutation(range(len(y)))
        return X[shuffled], y[shuffled]

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
