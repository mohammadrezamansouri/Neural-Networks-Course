from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np



(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

print("Training Data:", x_train.shape)
print("Test Data:", x_test.shape)



x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


network = models.Sequential()
network.add(layers.Flatten(input_shape=(32, 32, 3)))
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))



network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
network.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))


test_loss, test_acc = network.evaluate(x_test, y_test)

print("Test Accuracy:", test_acc)

for i in range(1, 10):
    plt.subplot(3, 3, i)
    index = np.random.randint(0, x_train.shape[0])  
    plt.imshow(x_train[index])  
    plt.axis('off')  

plt.show()