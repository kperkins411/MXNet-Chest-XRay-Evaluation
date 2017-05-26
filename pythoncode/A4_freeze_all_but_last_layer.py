#!/usr/bin/env python

import mxnet as mx
import settings
import numpy as np
import json
import logging
import A_utilities


###################
# UTILITIES
###################
# download a pretrained 50-layer ResNet model and load into memory.
# Note. If load_checkpoint reports error, we can remove the downloaded files and try get_model again.
import os, urllib


def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)


def get_model(prefix, epoch):
    download(prefix + '-symbol.json')
    download(prefix + '-%04d.params' % (epoch,))


# read/write a dict to a file
# used for serializing CNNcodes (output of headless CNN) for later training fully connected net
def writeDict(myDict, name):
    with open(name, "w") as outfile:
        json.dump(myDict, outfile, indent=4)


def readDict(name):
    try:
        with open(name, "r") as infile:
            dictValues = json.load(infile)
            return (dictValues)
    except IOError as e:
        print(e)
        raise
    except ValueError as e:
        print(e)
        raise
###################
# iterators
###################
def get_iterator(batch_size,
                 data_shape=(3, 224, 224),
                 path_imgrec=settings.RECORDIO_TRAIN_FILE,
                 data_name='data',
                 label_name='softmax_label',
                 shuffle=True,
                 rand_crop=False,
                 rand_mirror=False,
                 mean_img="../recordIO_dir/mean.bin"
                 ):
    #creates a data iterator from a recordIO file
    # first create the directory for lst and rec files
    # root = os.path.abspath(os.path.dirname(__file__))
    # recordIODir = os.path.join(root, settings.record_IO_directory)
    # train  = os.path.join(recordIODir, 'Train.rec')
    # val = os.path.join(recordIODir, 'Val.rec')

    return mx.io.ImageRecordIter(
        path_imgrec=path_imgrec,
        data_name=data_name,
        label_name=label_name,
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=shuffle,
        rand_crop=rand_crop,
        rand_mirror=rand_mirror,
        mean_img=mean_img)

######################################
# symbol and model manipulation
######################################
def get_part_of_symbol(symbol, layer_name):
    """
    a function which chops out all layers after layer_name

    symbol: the pre-trained network symbol
    layer_name: the layer name before the last fully-connected layer
    """
    # get the whole thing
    all_layers = symbol.get_internals()

    # create a new model that goes up to layer_name
    new_symbol = all_layers[layer_name]

    return (new_symbol)

def get_new_head(num_inputs, num_outputs):
    ''' creates a FC net
    :param num_inputs:  inputs to net
    :param num_outputs:   ooutputs from net
    :return: symbol
    '''
    new_head = mx.sym.Variable('data')
    new_head = mx.sym.FullyConnected(data=new_head, name='fc1', num_hidden=num_inputs)
    new_head = mx.sym.Activation(data=new_head, name='relu1', act_type="relu")
    new_head = mx.sym.FullyConnected(data=new_head, name='fc2', num_hidden=int ((num_inputs)/2) )
    new_head = mx.sym.Activation(data=new_head, name='relu2', act_type="relu")
    new_head = mx.sym.FullyConnected(data=new_head, name='fc3', num_hidden=int ((num_inputs)/4) )
    new_head = mx.sym.Activation(data=new_head, name='relu3', act_type="relu")
    new_head = mx.sym.FullyConnected(data=new_head, name='fc4', num_hidden=num_outputs)
    new_head = mx.sym.SoftmaxOutput(data=new_head, name='softmax')
    return new_head

def add_new_head(headless_symbol, num_output_classes):
    '''
    should go on after a relu layer

    tapes a 2 layer fully connected NN on to symbol which outputs num_output_classes
    :param headless_symbol:
    :param num_output_classes:
    :return:
    '''
    # # what is the size of the output for the headless_model
    # _, out_shape, _ = headless_symbol.infer_shape(data=(3, 224, 224))
    #
    # #get a head of the proper shape
    # new_head = get_new_head(out_shape, num_output_classes)

    #what is the size of the output of symbol
    new_symbol = mx.symbol.FullyConnected(data=headless_symbol, name='kp_fc1', num_hidden=512)
    new_symbol = mx.symbol.FullyConnected(data=new_symbol, name='kp_fc2', num_hidden=256)
    new_symbol = mx.symbol.FullyConnected(data=new_symbol, name='kp_fc3', num_hidden=128)
    new_symbol = mx.symbol.FullyConnected(data=new_symbol, name='kp_fc4', num_hidden=num_output_classes)
    new_symbol = mx.symbol.SoftmaxOutput(data=new_symbol, name='softmax')
    return new_symbol

######################################
# eval and train
######################################
def getHeadlessConvOutputs(mod, path_imgrec = settings.RECORDIO_TRAIN_FILE):
    '''

    Takes a pretrained partial model (hack off part of the end), and runs data from recIO file
    through it store models outputs and symbols
    Use to train a fully connected neural net

    :param mod: the hacked module
    :param path_imgrec: where recordio file located located
    :return:
    Outputs: list of mod outputs for given path_imgrec file
    Labels: corresponding labels
    '''

    Outputs = []
    Labels = []
    FILENAME =  path_imgrec.split('/')[-1].split('.')[0] +'.json'

    #if values already saved use them
    if  os.path.exists(FILENAME):
        dict = readDict(FILENAME)
        Outputs = dict["Outputs"]
        Labels = dict["Labels"]
    else:
        # create data iterator
        # I set the batch size to 1 so I would not have to deal with padding nor iterate over outputs
        BATCH_SIZE = 1
        CNNCodes_iter = get_iterator(batch_size = BATCH_SIZE,path_imgrec = path_imgrec)

        # iterate through and get the output of the module given the input
        # also collect the proper label
        for batch in CNNCodes_iter:
            mod.forward(batch, is_train=False)
            Outputs.append(mod.get_outputs()[0].asnumpy()[0].tolist())
            Labels.append(batch.label[0].asnumpy()[0].tolist())
        dict = {}
        dict["Outputs"] = Outputs
        dict["Labels"]  = Labels
        writeDict(dict, FILENAME)

    Outputs = np.asarray(Outputs)
    Labels = np.asarray(Labels)
    return Outputs,Labels

def create_CNN_codes(CnnCodes, arg_params,aux_params):
    '''
    :param CnnCodes: symbol. orig conv net with some layer (probably just FC one) removed
    :param arg_params: orig conv net weights, biases
    :param aux_params: orig conv net batch norm params

    '''
    # create a module using CNNCodes only
    orig_CNN_mod = mx.mod.Module(symbol=CnnCodes, label_names=None, context=mx.gpu())
    orig_CNN_mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])

    # init params, including the new net
    orig_CNN_mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
                             force_init=True)
    # then set the orig net to orig params
    orig_CNN_mod.set_params(arg_params, aux_params, allow_missing=True, force_init=True)

    # get what the headless conv outputs, and the correct labels
    train_outputs, train_labels = getHeadlessConvOutputs(orig_CNN_mod, path_imgrec=settings.RECORDIO_TRAIN_FILE)
    test_outputs, test_labels = getHeadlessConvOutputs(orig_CNN_mod, path_imgrec=settings.RECORDIO_TEST_FILE)
    return(train_outputs, train_labels,test_outputs, test_labels)

def create_and_train_FC_net(train_outputs, train_labels,test_outputs, test_labels, batch_size, num_classes, num_epoch):
    '''
    creates and trains a FC net using CNNCodes as inputs, will save the best model in epoch_end_callback
    :param train_outputs:
    :param train_labels:
    :param test_outputs:
    :param test_labels:
    :param batch_size:
    :param num_classes:
    :param num_epoch:
    :return: nothing, fit has an epoch end callback to check if we have best validation score, if so this FC networks
            params are saved to A_utilities.paramFile
    '''



    # create  iterators from above data
    FC_Training_iter = mx.io.NDArrayIter(data=train_outputs, label=train_labels, batch_size=batch_size, shuffle=True)
    FC_Test_iter = mx.io.NDArrayIter(data=test_outputs, label=test_labels, batch_size=batch_size, shuffle=True)

     # create a stand alone FC net to train on above iterators
    new_head_symbol = get_new_head(num_inputs=train_outputs.shape[1], num_outputs=num_classes)
    model = mx.mod.Module(context=mx.gpu(), symbol=new_head_symbol)

    # train the FC net, saving the best model
    model.fit(train_data=FC_Training_iter, eval_data=FC_Test_iter, eval_metric='acc',
              batch_end_callback=mx.callback.Speedometer(batch_size=batch_size, frequent=100),
              epoch_end_callback=A_utilities.epoc_end_callback_kp, optimizer='sgd',
              optimizer_params={'learning_rate': 0.01, 'momentum': 0.9}, num_epoch=num_epoch, force_rebind=True,
              initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))


def main():
    num_epoch = 1000
    num_classes = 2
    batch_per_gpu = 16
    num_gpus = 1
    batch_size = batch_per_gpu * num_gpus

    best_FC_modelname = "BEST_FC_MODEL"
    best_CNN_FC_modelname = "BEST_CNN_FC_MODEL"

    train_FC = False
    train_CNN_FC = False

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    # mx.profiler.profiler_set_config(mode='all', filename='profile_output.json')
    # mx.profiler.profiler_set_state('run')

    #load model
    get_model('http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN', 126)
    sym, arg_params, aux_params = mx.model.load_checkpoint('Inception-BN', 126)

    ################### CNNCODES and FC head training
    # get partial network, exclude last FC layer
    # look at *.json file downloaded with the model above to see what the layer names are
    CnnCodes = get_part_of_symbol(sym, layer_name="flatten_output")

    #strip out params that only apply to CnnCodes
    arg_params, aux_params = A_utilities.strip_obsolete_params(arg_params = arg_params,aux_params=aux_params ,
                                                               new_symbol=CnnCodes)

    #use old stuff?
    if train_FC == True or os.path.exists(best_FC_modelname+'-00002.params'):
        #get outputs of headless CNN usong Train and Val dataset
        train_outputs, train_labels, test_outputs, test_labels = create_CNN_codes(CnnCodes, arg_params,aux_params)

        # where will best model be written?
        A_utilities.reset_epoch_end_callback(best_FC_modelname)

        #now train FC net on above outputs, save best model to
        create_and_train_FC_net(train_outputs, train_labels, test_outputs, test_labels,batch_size, num_classes, num_epoch)

    ################### combine headless trained CNN and trained FC net

    #get train iterators
    train_itr = get_iterator(batch_size, path_imgrec=settings.RECORDIO_TRAIN_FILE)
    test_itr = get_iterator(batch_size, path_imgrec=settings.RECORDIO_TEST_FILE)

    #now load the best FC layer from file system
    fc_sym, fc_arg_params, fc_aux_params = mx.model.load_checkpoint(best_FC_modelname, 2)

    #then tape FC net to headless Conv net
    cnn_plus_newhead = fc_sym(data = CnnCodes,name= "cnn_plus_newhead")

    #create model
    cnn_plus_newhead_mod = mx.mod.Module(symbol=cnn_plus_newhead,  context=mx.gpu())

    #bind
    cnn_plus_newhead_mod.bind(for_training=True, data_shapes=train_itr.provide_data, label_shapes=train_itr.provide_label)

    #set params CNN part then FC part
    cnn_plus_newhead_mod.set_params(arg_params, aux_params, allow_missing=True, force_init=False)
    cnn_plus_newhead_mod.set_params(fc_arg_params, fc_aux_params, allow_missing=True, force_init=True)

    ################### now end to end training on almost there net
    # use old stuff?
    if train_CNN_FC == True or os.path.exists(best_CNN_FC_modelname + '-00002.params'):
        # where will results be written?
        A_utilities.reset_epoch_end_callback(best_CNN_FC_modelname)

        #train end to end model
        cnn_plus_newhead_mod.fit(train_data=train_itr, eval_data=test_itr, eval_metric='acc',
                  batch_end_callback=mx.callback.Speedometer(batch_size=batch_size, frequent=100),
                  epoch_end_callback=A_utilities.epoc_end_callback_kp, optimizer='sgd',
                  optimizer_params={'learning_rate': 0.01, 'momentum': 0.9}, num_epoch=num_epoch)

    ################### now lets see how it works on validation dataset
    ##BTW the following is what you want to use if you put this on the web
    #except you want to predict 1 image at a time

    # now load the best end to end model from file system
    cnn_fc_sym, cnn_fc_arg_params, cnn_fc_aux_params = mx.model.load_checkpoint(best_CNN_FC_modelname, 2)

    #get iterator
    val_itr = get_iterator(batch_size, path_imgrec=settings.RECORDIO_VAL_FILE)

    # create model
    cnn_plus_newhead_mod = mx.mod.Module(symbol=cnn_fc_sym, context=mx.gpu())

    # bind
    cnn_plus_newhead_mod.bind(for_training=False, data_shapes=test_itr.provide_data,
                              label_shapes=test_itr.provide_label)

    # set params
    cnn_plus_newhead_mod.set_params(cnn_fc_arg_params, cnn_fc_aux_params,allow_missing=False,force_init=True)

    #the raw softmax outputs
    outputs = cnn_plus_newhead_mod.predict(test_itr, ['mse', 'acc']).asnumpy()

    score = cnn_plus_newhead_mod.score(test_itr, ['mse', 'acc', 'F1'])
    print score

if __name__ == '__main__':
    main()