from keras.models import Sequential
from keras import regularizers
from keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
)

# https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
def cnn_model(input_shape):
    weight_decay = 1e-4
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
            input_shape=input_shape,
        )
    )
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)
        )
    )
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(
        Conv2D(
            64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)
        )
    )
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(weight_decay)
        )
    )
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
    )
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
    )
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    model.summary()
    return model


if __name__ == "__main__":
    input_shape = (32, 32, 3)
    cnn_model(input_shape)
