# Tensorflow instances

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Sequential

# Shallow Model
# Build CNN model

def complete_model(input_shape:tuple, num_classes:int)->Sequential:
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
    A basic CNN.
    """

    model = Sequential([

        Conv2D(32,3,padding='same',input_shape=input_shape, activation='relu'),
        BatchNormalization(),

        Conv2D(64,3,padding='same', activation='relu'),
        BatchNormalization(),

        Conv2D(128,3,padding='same', activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        Dropout(0.2),

    # Add a classifier on top of the CNN
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    return model