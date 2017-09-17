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
parser.add_argument('--network', type = str, default = 'cold_plains_v1', help = 'Which type of network to use to train',
                    choices = ['cold_plains_v1', 'cold_plains_v2', 'cold_plains_v3'])

args = parser.parse_args()

if not os.path.exists(args.outputFolder):
    os.mkdir(args.outputFolder)

if args.training is None:
    print "Please specify a training dataset"
    exit(-1)

with h5py.File(args.training, 'r') as f:
    Xshape = f['X'].shape
    framesShape = f['frames'].shape
    rewards_ = f['rewards'][:].flatten()

if args.rewards:
    with h5py.File(args.rewards, 'r') as f:
        rewards_ = f['rewards'][:].flatten()
    
if args.input is not None:
    model = keras.models.load_model(args.input)

    if args.fineTune > 0:
        L = len(model.layers)
        for i in range(L - args.fineTune):
            model.layers[i].trainable = False
else:
    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Input, Concatenate, AveragePooling2D
    from keras import regularizers, losses
    from keras.optimizers import Adam

    if args.network == 'cold_plains_v1':
        inp1 = Input(shape = (260, 260, 3))
        c1 = Conv2D(8, kernel_size=(3, 3), activation='relu')(inp1)
        c2 = Conv2D(12, (3, 3), activation='relu')(c1)
        max1 = MaxPooling2D(pool_size = (4, 4))(c2)
        c3 = Conv2D(16, kernel_size=(3, 3), activation='relu')(max1)
        c4 = Conv2D(20, (3, 3), activation='relu')(c3)
        max3 = MaxPooling2D(pool_size = (4, 4))(c4)
        drop1 = Dropout(0.25)(max3)
        flatten = Flatten()(drop1)
        denseCnn = Dense(128, activation='relu')(flatten)
        drop2 = Dropout(0.5)(denseCnn)
    elif args.network == 'cold_plains_v2':
        inp1 = Input(shape = (400, 640, 3))
        AveragePooling2D(pool_size = (2, 2))
        c1 = Conv2D(8, kernel_size = (3, 3), strides = (2, 2), activation='relu')(inp1)
        c2 = Conv2D(12, kernel_size = (3, 3), strides = (2, 2), activation='relu')(c1)
        c3 = Conv2D(16, kernel_size = (3, 3), strides = (2, 2), activation='relu')(c2)
        c4 = Conv2D(20, kernel_size = (3, 3), strides = (2, 2), activation='relu')(c3)
        #drop1 = Dropout(0.25)(c4)
        flatten = Flatten()(c4)
        denseCnn = Dense(128, activation='relu')(flatten)
        drop2 = Dropout(0.5)(denseCnn)
    else:
        inp1 = Input(shape = (400, 640, 3))
        c1 = Conv2D(4, strides=(2, 2), kernel_size=(3, 3), activation='relu')(inp1)
        c2 = Conv2D(8, (3, 3), activation='relu')(c1)
        max1 = MaxPooling2D(pool_size = (4, 4))(c2)
        c3 = Conv2D(12, kernel_size=(3, 3), activation='relu')(max1)
        c4 = Conv2D(16, (3, 3), activation='relu')(c3)
        max3 = MaxPooling2D(pool_size = (4, 4))(c4)
        c5 = Conv2D(20, kernel_size=(3, 3), activation='relu')(max3)
        max4 = MaxPooling2D(pool_size = (2, 2))(c5)
        drop1 = Dropout(0.25)(max4)
        flatten = Flatten()(drop1)
        denseCnn = Dense(128, activation='relu')(flatten)
        drop2 = Dropout(0.5)(denseCnn)

    inp2 = Input(shape = [8])
    dense3 = Dense(15, activation = 'relu')(inp2)
    concat = Concatenate()([drop2, dense3])
    dense4 = Dense(32, activation = 'relu')(concat)
    dense5 = Dense(16, activation = 'relu')(dense4)
    out = Dense(1)(dense4)
    model = Model(inputs = [inp1, inp2], outputs = out)
    
model.compile(optimizer = 'adam',#Adam(lr = 0.01),
              loss = 'mse')#losses.mean_absolute_error)#

model.summary()
    
def generator(filename, batchsize):
    with h5py.File(filename, 'r') as f:
        N = f['X'].shape[0]

        i = 0
        while 1:
            ir = i + batchsize
            if ir > N:
                i = 0
                continue
            
            X = f['X'][i : ir].reshape(-1, 8)
            frames = f['frames'][i : ir, :, :, :]
            rewards = rewards_[i : ir]#f['rewards'][i : ir].flatten()

            i = i + batchsize

            if i > N:
                i = 0

            yield ([frames, X], rewards.flatten())
    #Xshape = f['X']
    #frames_shape = f
    #frames = f['frames'][:]
    #rewards = f['rewards'][:]

#frames = frames[:, 
with h5py.File(args.training, 'r') as f:
    X = f['X'][:2000].reshape(-1, 8)
    frames = f['frames'][:2000, :, :, :]
    rewards = rewards_[:2000]#f['rewards'][:2000]

open("{0}/loss".format(args.outputFolder), "w").close()

class LossHistory(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        with open("{0}/loss".format(args.outputFolder), "a") as f:
            f.write("{0}\n".format(logs.get('loss')))

history = LossHistory()

for i in range((args.epochs + args.checkpoint - 1) / args.checkpoint):
    epochs = min(args.epochs - i * args.checkpoint, args.checkpoint)
    print "Checkpoint {0}, epochs {1} to {2}".format(i, i * args.checkpoint, min(args.epochs, i * args.checkpoint))
    model.fit_generator(generator(args.training, args.batchsize),
                        (Xshape[0] + args.batchsize - 1) / args.batchsize,
                        epochs = epochs,
                        callbacks = [history])
    prewards = model.predict([frames, X])
    plt.plot(prewards, rewards, '*')
    plt.plot(rewards, rewards, 'r')
    plt.savefig('{0}/plot{1}.png'.format(args.outputFolder, i))
    plt.clf()

    if args.output:
        model.save(args.output)
