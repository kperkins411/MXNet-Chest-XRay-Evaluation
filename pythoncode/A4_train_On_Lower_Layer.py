#!/usr/bin/env python

import mxnet as mx
import settings
import numpy as np


def get_iterator(batch_size,
                 data_shape=(3, 224, 224),
                 path_imgrec='../recordIO_dir/Train.rec',
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

#define the function which returns the data iterators.
def get_iterators(batch_size, data_shape=(3, 224, 224)):
    # first create the directory for lst and rec files
    root = os.path.abspath(os.path.dirname(__file__))
    recordIODir = os.path.join(root, settings.record_IO_directory)
    train  = os.path.join(recordIODir, 'Train.rec')
    val = os.path.join(recordIODir, 'Val.rec')

    train = mx.io.ImageRecordIter(
        path_imgrec         = '../recordIO_dir/Train.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True,
        mean_img           = "../recordIO_dir/mean.bin")
    val = mx.io.ImageRecordIter(
        path_imgrec         = '../recordIO_dir/Val.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False,
        mean_img="../recordIO_dir/mean.bin")
    test = mx.io.ImageRecordIter(
        path_imgrec='../recordIO_dir/Test.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=False,
        rand_mirror=False,
        mean_img="../recordIO_dir/mean.bin")
    return (train, val,test)


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
    '''
    :param num_inputs:
    :param num_outputs:
    :return:
    '''
    new_head = mx.sym.Variable('data')
    new_head = mx.sym.FullyConnected(data=new_head, name='fc1', num_hidden=num_inputs)
    new_head = mx.sym.Activation(data=new_head, name='relu1', act_type="relu")
    new_head = mx.sym.FullyConnected(data=new_head, name='fc2', num_hidden=int ((num_inputs-num_outputs)/2) )
    new_head = mx.sym.Activation(data=new_head, name='relu2', act_type="relu")
    new_head = mx.sym.FullyConnected(data=new_head, name='fc3', num_hidden=num_outputs)
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


def get_module_outputs_and_labels(mod, itr):
    '''
    Takes a pretrained partial model (hack off part of the end), and runs data from recIO file
    through it store models outputs and symbols
    Use to train a fully connected neural net

    :param mod: the hacked module
    :param itr: an iterator for the data
    :return:
    '''
    # a lot of the below came from base_module.predict
    # I set the batch size to 1 so I would not have to deal with padding nor iterate over outputs

    # get an iterator
    # itr = get_iterator(batch_size=1, path_imgrec = path_imgrec )

    # where results go
    outputs = []
    labels = []

    itr.reset()

    #iterate through and get the output of the module given the input
    #also collect the proper label
    for eval_batch in itr:
        # what is this batch supposed to be
        label = eval_batch.label[0].asnumpy()[0]
        labels.append(label)

        # same as the following
        mod.forward(eval_batch, is_train=False)
        out = mod.get_outputs()[0].asnumpy()[0] #returns output on GPU
        # outcpu = out.as_in_context(eval_batch.context)
        # toutputs = type(out)
        # out = np.asarray(mod.get_outputs())
        # out2= out1.asnumpy()
        #The following does the entire iterator
        # out = mod.predict(itr, merge_batches=False)
        outputs.append(out)

    # outputs2 = mod.predict(itr, merge_batches=False)
    return np.asarray(outputs),np.asarray(labels)



# download a pretrained 50-layer ResNet model and load into memory.
# Note. If load_checkpoint reports error, we can remove the downloaded files and try get_model again.
import os, urllib
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)
def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

####################################################################################

import logging
import A_utilities

def main():
    batch_size = 8
    num_epoch = 100

    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(level=logging.DEBUG, format=head)

    get_model('http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN', 126)
    sym, arg_params, aux_params = mx.model.load_checkpoint('Inception-BN', 126)

    attr_dict_sym = sym.attr_dict()

    # first we will get a non shuffalable iterator
    # feed it through our resnet fragment
    # and get the results,
    # we will use this to train a NN
    num_classes = 2
    batch_per_gpu = 1
    num_gpus = 1
    batch_size = batch_per_gpu * num_gpus

    # todo Change this back to train for dev
    train_itr = get_iterator(batch_size, path_imgrec='../recordIO_dir/Train.rec')
    val_itr = get_iterator(batch_size, path_imgrec='../recordIO_dir/Val.rec')

    # get a partial network
    # first lets get up to the second to the last relu
    # look at the http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-symbol.json
    # downloaded with the model above to see what the layer names are
    #newsymbol = get_part_of_symbol(sym, layer_name="stage4_unit2_relu1_output")
    newsymbol = get_part_of_symbol(sym, layer_name="flatten_output")

    # now add a new head to it
    newsymbol = add_new_head(newsymbol, num_classes)

    #get a dict of attributes in new model
    attr_dict_newsymbol = newsymbol.attr_dict()

    #find diffs and output to console
    # A_utilities.output_diffs(attr_dict_newsymbol, attr_dict_sym, "attr_dict_newsymbol" ,"attr_dict_sym" )
    # A_utilities.output_diffs(arg_params, attr_dict_newsymbol, "arg_params","attr_dict_newsymbol")
    # A_utilities.output_diffs(aux_params, attr_dict_sym, "aux_params", "attr_dict_sym")

    #whats in arg and aux params pre
    A_utilities.output_diffs(arg_params, newsymbol.list_arguments(), "arg_params", "newsymbol.list_arguments()")
    A_utilities.output_diffs(aux_params, attr_dict_newsymbol,"aux_params", "attr_dict_newsymbol")

    #strip out params that only apply to old model
    arg_params, aux_params = A_utilities.strip_obsolete_params(arg_params = arg_params,aux_params=aux_params , new_symbol=newsymbol)

    #params stripped, no extras in aux or arg params
    A_utilities.output_diffs(arg_params, newsymbol.list_arguments(), "arg_params", "newsymbol.list_arguments()")
    A_utilities.output_diffs(aux_params, attr_dict_newsymbol,"aux_params", "attr_dict_newsymbol")

    # create a module using new network
    mod = mx.mod.Module(symbol=newsymbol, context=mx.gpu())
    mod.bind(for_training=True, data_shapes=train_itr.provide_data, label_shapes=train_itr.provide_label)

    # init params, including the new net
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),force_init=True)

    # then set the orig net to orig params
    mod.set_params(arg_params, aux_params, allow_missing=True, force_init=True)
    ########################################################################################################################

    # train_inputs, train_labels = get_module_outputs_and_labels(mod, train_itr)
    # val_inputs, val_labels = get_module_outputs_and_labels(mod, val_itr)
    #
    #
    # new_train_itr = mx.io.NDArrayIter(data=train_inputs, label=np.asarray(train_labels), batch_size=batch_size,
    #                                   last_batch_handle='roll_over')
    # new_val_itr = mx.io.NDArrayIter(data=val_inputs, label=np.asarray(val_labels), batch_size=batch_size,
    #                                 last_batch_handle='roll_over')
    # print("created new itertors to train new FC network on")
    # mod.fit(train_data=new_train_itr, eval_data=new_val_itr, eval_metric='acc',
    mod.fit(train_data=train_itr,  eval_data=val_itr, eval_metric='acc',
        batch_end_callback=mx.callback.Speedometer(batch_size=batch_size, frequent=100),  epoch_end_callback=A_utilities.epoc_end_callback_kp, optimizer='sgd',
        optimizer_params={'learning_rate': 0.001, 'momentum': 0.9}, num_epoch=num_epoch)

if __name__ == '__main__':
    main()