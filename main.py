import argparse, os
import numpy as np

import pandas as pd
import numpy as np
import pickle
from scipy import stats
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import backend as K


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default='/tmp')
    parser.add_argument('--training', type=str, default='data')
    parser.add_argument('--validation', type=str, default='data')
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    RANDOM_SEED = 42
    LABELS = ["Normal", "Fraud"]
    
    # Download the data set
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, engine="python") for file in input_files ]
    df = pd.concat(raw_data)
    #print(df.head())
    #data = df.drop(['Time'], axis=1)

    #data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = df
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis=1)

    y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis=1)

    X_train = X_train.values
    X_test = X_test.values
    print(X_train.shape)
    
    
    input_dim = X_train.shape[1]
    encoding_dim = 14
    
    
    input_layer = Input(shape=(input_dim, ))

    encoder = Dense(encoding_dim, activation="tanh", 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    nb_epoch = 2
    batch_size = 32

    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="model.h5",
                                   verbose=0,
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        verbose=1,
                        callbacks=[checkpointer, tensorboard]).history

    autoencoder = load_model('model.h5')
    
    
    
    
    
    # save Keras model for Tensorflow Serving
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(args.sm_model_dir, '1'),
        inputs={'inputs': autoencoder.input},
        outputs={t.name: t for t in autoencoder.outputs})
    
    
#     # labels are in the first column
#     train_y = train_data.ix[:,30]
#     train_X = train_data.ix[:,:30]
    
#     df = pd.read_csv("data/creditcard.csv")
    
    
    
#     os.makedirs("./data", exist_ok = True)
#     (x_train, y_train), (x_val, y_val) = fashion_mnist.load_data()
#     np.savez('./data/training', image=x_train, label=y_train)
#     np.savez('./data/validation', image=x_val, label=y_val)
    
#     x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
#     y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
#     x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
#     y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']
    
#     # input image dimensions
#     img_rows, img_cols = 28, 28

#     # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
#     print(K.image_data_format())
#     if K.image_data_format() == 'channels_first':
#         print("Incorrect configuration: Tensorflow needs channels_last")
#     else:
#         # channels last
#         x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#         x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
#         input_shape = (img_rows, img_cols, 1)
#         batch_norm_axis=-1

#     print('x_train shape:', x_train.shape)
#     print(x_train.shape[0], 'train samples')
#     print(x_val.shape[0], 'test samples')
    
#     # Normalize pixel values
#     x_train  = x_train.astype('float32')
#     x_val    = x_val.astype('float32')
#     x_train /= 255
#     x_val   /= 255
    
#     # Convert class vectors to binary class matrices
#     num_classes = 10
#     y_train = keras.utils.to_categorical(y_train, num_classes)
#     y_val   = keras.utils.to_categorical(y_val, num_classes)
    
#     model = Sequential()
    
#     # 1st convolution block
#     model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=input_shape))
#     model.add(BatchNormalization(axis=batch_norm_axis))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    
#     # 2nd convolution block
#     model.add(Conv2D(128, kernel_size=(3,3), padding='valid'))
#     model.add(BatchNormalization(axis=batch_norm_axis))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=2))

#     # Fully connected block
#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.3))

#     # Output layer
#     model.add(Dense(num_classes, activation='softmax'))
    
#     print(model.summary())

#     if gpu_count > 1:
#         model = multi_gpu_model(model, gpus=gpu_count)
        
#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
#                   metrics=['accuracy'])
    
#     model.fit(x_train, y_train, 
#               batch_size=batch_size,
#               validation_data=(x_val, y_val), 
#               epochs=epochs,
#               verbose=1)
    
#     score = model.evaluate(x_val, y_val, verbose=0)
#     print('Validation loss    :', score[0])
#     print('Validation accuracy:', score[1])
    
#     # save Keras model for Tensorflow Serving
#     sess = K.get_session()
#     tf.saved_model.simple_save(
#         sess,
#         os.path.join(model_dir, 'model/1'),
#         inputs={'inputs': model.input},
#         outputs={t.name: t for t in model.outputs})
    
