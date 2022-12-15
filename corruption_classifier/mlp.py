import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam


class MLP:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.model = Sequential()
        self.model.add(Dense(hidden_dim, activation='relu', input_dim=512))
        self.model.add(Dense(4, activation='softmax'))
        self.model.compile(loss='SparseCategoricalCrossentropy',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy'])
    def fit(self, X, y):
        self.model.fit(X, y, epochs=1, batch_size=128, verbose=0)

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def score(self, X, y):
        return self.model.evaluate(X, y, batch_size=128, verbose=0)[1]
       