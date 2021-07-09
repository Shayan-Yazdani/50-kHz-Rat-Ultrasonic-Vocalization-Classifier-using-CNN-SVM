
from tensorflow.python.keras import layers, Input, Model
from tensorflow.python.keras import Sequential

def DenoisingAutoencoder():

    autoencoder = Sequential()

    #Encoder
    autoencoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(224, 224, 1)))
    autoencoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.MaxPooling2D((2, 2), padding='same'))

    #Decoder
    autoencoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.UpSampling2D((2, 2)))
    autoencoder.add(layers.Conv2D(32, (3, 3), activation='tanh', padding='same'))
    autoencoder.add(layers.UpSampling2D((2, 2)))
    autoencoder.add(layers.Conv2D(1, (3, 3), activation='tanh', padding='same'))
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder
