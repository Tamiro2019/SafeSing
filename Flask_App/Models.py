from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from Signal_Processing import *

def load_model():
    im_f= 1025
    im_t= 44
    n_samples = 909

    num_filters = 8
    filter_size = 3
    pool_size = 2

    # Build the model.
    model = Sequential([
      Conv2D(num_filters, filter_size, input_shape=(im_f, im_t, 1)),
      MaxPooling2D(pool_size=pool_size),
      Flatten(),
      Dense(3, activation='softmax'),
    ])

    # Compile the model.
    model.compile(
      'adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )

    # Load pre-trained weights

    model.load_weights('ssing_cnn_v1.h5')

    return model

# predictor function
def model_prediction(model,wave):
    '''
    Expects raw wave from librosa.load(filename,sr=44100)
    '''
    #model = load_model()

    RdB_ready = audio_processor(wave, sr=44100, order=16)
    predict = model.predict(RdB_ready)
    prediction = np.argmax(predict, axis=1)[0]
    return prediction
