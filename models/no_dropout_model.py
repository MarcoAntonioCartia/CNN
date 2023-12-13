# Tensorflow instances

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Sequential

# Shallow Model
# Build CNN model

def no_dropout_model(input_shape:tuple, num_classes:int)->Sequential:
    """
    Parameters
    ----------
    input_shape : tuple 
    num_classes : int 

    Returns
    ----------
    Sequential

    Notes
    ----------
    A basic CNN with no dropout in their weights.
    """

    model = Sequential([

        Conv2D(32,3,padding='same',input_shape=(32,32,3), activation='relu'),
        BatchNormalization(),

        Conv2D(64,3,padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(128,3,padding='same', activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),

    # Add a classifier on top of the CNN
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    return model