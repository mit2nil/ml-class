import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.activations import relu
from keras.layers import Dense, Flatten, Dropout, Reshape, Conv2D, MaxPooling2D, Activation
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.epochs = 20
config.lr = 0.01
config.layers = 3
config.hidden_layer_1_size = 128

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.
X_test = X_test / 255.

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels =["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

my_activation = Activation(lambda x: relu(x, alpha=0.1))

# create model
model=Sequential()
model.add(Reshape((img_width, img_height, 1), input_shape=(img_width,img_height)))
model.add(Conv2D(54,(4,4),kernel_initializer='glorot_uniform', bias_initializer='zeros',input_shape=(28, 28,1),activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(9,(2,2),kernel_initializer='glorot_uniform', bias_initializer='zeros',input_shape=(28, 28,1),activation="linear", padding="same"))
model.add(my_activation)
model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Dropout(0.45))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=labels)])

