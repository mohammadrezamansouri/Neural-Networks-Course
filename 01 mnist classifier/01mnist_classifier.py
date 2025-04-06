
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


#  DATA MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

'''The codes are written in the same way as the class notes.'''


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = Sequential([

    Flatten(input_shape=(28, 28)),

    Dense(200, activation='relu'),

    Dense(120, activation='relu'),
   
    Dense(10, activation='softmax')
])


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


test_loss, test_acc = model.evaluate(x_test, y_test)
print("test acc :", test_acc)
