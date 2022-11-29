import numpy as np
from keras.models import Sequential
from keras.layers import Dense

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]], dtype=np.float32)
Y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=100, verbose=2, batch_size=1)
print(model.summary())
print(model.predict(X))