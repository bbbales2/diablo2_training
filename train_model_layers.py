import os
import argparse
import numpy
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import keras

parser = argparse.ArgumentParser(description='Train neural network for diablo2')
parser.add_argument('training', type = str, default = None, help = 'hdf5 file with training data')
parser.add_argument('--rewards', type = str, default = None, help = 'Optional hdf5 file with separate rewards')
parser.add_argument('--input', type = str, default = None, help = 'Input model to continue training')
parser.add_argument('--fineTune', type = int, default = -1, help = 'Number of layers back from the end to fine tune. -1 means train all layers')
parser.add_argument('--output', type = str, default = None, help = 'Output model name. By default don\'t save models')
parser.add_argument('--epochs', type = int, default = 1000, help = 'Number of training epochs')
parser.add_argument('--batchsize', type = int, default = 100, help = 'Size of batches')
parser.add_argument('--checkpoint', type = int, default = 10, help = 'Save the model ever checkpoint epochs')
parser.add_argument('--outputFolder', type = str, default = 'output', help = 'Output folder to put diagnostics plots in. By default output')
parser.add_argument('--network', type = str, default = 'layer_v1', help = 'Which type of network to use to train',
                    choices = ['layer_v1'])

args = parser.parse_args()

if not os.path.exists(args.outputFolder):
    os.mkdir(args.outputFolder)

if args.training is None:
    print "Please specify a training dataset"
    exit(-1)

with h5py.File(args.training, 'r') as f:
    clickHistories = f['clickHistories'][:]
    clicks = f['clicks'][:]
    missingHp = f['missingHp'][:]
    layers = f['layers'][:]
    rewards = f['rewards'][:]

if args.rewards:
    with h5py.File(args.rewards, 'r') as f:
        rewards = f['rewards'][:].flatten()
    
if args.input is not None:
    model = keras.models.load_model(args.input)

    if args.fineTune > 0:
        L = len(model.layers)
        for i in range(L - args.fineTune):
            model.layers[i].trainable = False
else:
    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, Input, Concatenate, AveragePooling2D, LSTM, GRU
    from keras import regularizers, losses
    from keras.optimizers import Adam, Adagrad, Adadelta

    #model = Sequential()
    #model.add(GRU(128, return_sequences=True, input_shape=(5, layers[0].shape[0] + clicks[0].shape[0])))
    #model.add(GRU(64, return_sequences=True))
    #model.add(GRU(32))
    #model.add(Dense(1))

    inp_layer = Input(shape = (1, layers[0].shape[0]))
    click_layer = Input(shape = (1, clicks[0].shape[0]))
    dense1 = Dense(128, activation = 'relu')(Flatten()(inp_layer))
    drop_layer = Dropout(0.5)(dense1)
    dense_click = Dense(15, activation = 'relu')(Flatten()(click_layer))
    concat = Concatenate()([drop_layer, dense_click])
    dense3 = Dense(32, activation = 'relu')(concat)
    dense4 = Dense(16, activation = 'relu')(dense3)
    #gru1 = GRU(64, return_sequences=True)(concat)
    #gru2 = GRU(32, return_sequences=True)(gru1)
    #gru3 = GRU(8)(gru2)
    out = Dense(1, activation = 'relu')(dense4)
    
    model = Model(inputs = [inp_layer, click_layer], outputs = out)

    #if args.network == 'layer_v1':
    #    inp_layer = Input(shape = layers[0].shape)
    #    denseCnn = Dense(128, activation='relu')(inp_layer)
    #    drop_layer = Dropout(0.5)(denseCnn)

    #inp_click_history = Input(shape = clickHistories[0].shape)
    #flatten_click_history = Flatten()(inp_click_history)
    #dense_click_history1 = Dense(15, activation = 'relu')(flatten_click_history)
    #dense_click_history2 = Dense(8, activation = 'relu')(dense_click_history1)

    #inp_hp = Input(shape = (1,))
    #dense_hp = Dense(4, activation = 'relu')(inp_hp)

    #inp_clicks = Input(shape = clicks[0].shape)
    #dense_clicks = Dense(15, activation = 'relu')(inp_clicks)
    #concat = Concatenate()([drop_layer, dense_clicks, dense_hp, dense_click_history2])
    #dense4 = Dense(32, activation = 'relu')(concat)
    #dense5 = Dense(16, activation = 'relu')(dense4)
    #out = Dense(1)(dense4)
    #model = Model(inputs = [inp_layer, inp_click_history, inp_clicks, inp_hp], outputs = out)
    
model.compile(optimizer = Adadelta(),#m(lr = 0.01),#'adam',
              loss = 'mse')

model.summary()

open("{0}/loss".format(args.outputFolder), "w").close()

class LossHistory(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        with open("{0}/loss".format(args.outputFolder), "a") as f:
            f.write("{0}\n".format(logs.get('loss')))

history = LossHistory()

T = 1
X = numpy.concatenate([layers, clicks], axis = 1)
X_ = numpy.zeros((X.shape[0], T, X.shape[1]))
for i in range(T, X_.shape[0]):
    for j in range(T):
        X_[i, j, :] = X[i - T + j + 1, :]
X_ = X_[T:]

for i in range((args.epochs + args.checkpoint - 1) / args.checkpoint):
    epochs = min(args.epochs - i * args.checkpoint, args.checkpoint)
    print "Checkpoint {0}, epochs {1} to {2}".format(i, i * args.checkpoint, min(args.epochs, i * args.checkpoint))
    print X_.shape
    model.fit([X_[:, :, :640], X_[:, :, 640:]],#clickHistories, , missingHp
              rewards[T:],
              batch_size = args.batchsize,
              epochs = epochs,
              callbacks = [history])
    prewards = model.predict([X_[:5000, :, :640], X_[:5000, :, 640:]])#,[layers[:5000], clicks[:5000]] missingHp[:5000], clickHistories[:5000]
    plt.plot(prewards, rewards[T:][:5000] + numpy.random.randn(len(prewards)) * 0.1, '.', alpha = 0.1)
    plt.plot(rewards, rewards, 'r')
    plt.savefig('{0}/plot{1}.png'.format(args.outputFolder, i))
    plt.clf()

    if args.output:
        model.save(args.output)
