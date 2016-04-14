from keras.models import Sequential
import input_data
from keras.layers import Dense, Dropout, Activation,Merge,LSTM
from keras.utils.np_utils import to_categorical
model = Sequential()
model.add(Dense(512,input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

data,label=mnist.train.next_batch(55000)
X_test,Y_test=mnist.test.next_batch(1000)

model.fit(data, label, nb_epoch=40, batch_size=32,verbose=1, validation_data=(X_test, Y_test))
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
print loss_and_metrics