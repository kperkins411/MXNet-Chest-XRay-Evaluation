import mxnet as mx


#create train and val iterators
import early_stopping as es
train_val,train_lab= es.getBinaryArrayAndLabels(numberrows=es.numb_Training_samples, numbBinDigits=es.numb_bin_digits)
itr_train=mx.io.NDArrayIter(train_val,train_lab,batch_size=es.batch_size,shuffle='True')

#get eval iterators
eval_val,eval_lab =es.getBinaryArrayAndLabels(numberrows=es.numb_Validation_samples, numbBinDigits=es.numb_bin_digits)
itr_val = mx.io.NDArrayIter(eval_val,eval_lab,batch_size=es.batch_size)

num_inputs =es.numb_bin_digits
num_outputs = es.numb_softmax_outputs
numb_hidden = 100

#create a symbolic net
new_head = mx.sym.Variable('data')
new_head = mx.sym.FullyConnected(data=new_head, name='fc1', num_hidden=num_inputs)
new_head = mx.sym.Activation(data=new_head, name='relu1', act_type="relu")
new_head = mx.sym.FullyConnected(data=new_head, name='fc2', num_hidden=numb_hidden)
new_head = mx.sym.Activation(data=new_head, name='relu2', act_type="relu")
new_head = mx.sym.FullyConnected(data=new_head, name='fc3', num_hidden=num_outputs)
new_head = mx.sym.SoftmaxOutput(data=new_head, name='softmax')

all_layers = new_head.get_internals()

#now lets swap out part of it
new_symbol = all_layers["relu1_output"]
new_symbol = mx.sym.Activation(data=new_symbol, name='relu3', act_type="relu")
new_symbol = mx.sym.FullyConnected(data=new_symbol, name='KP_fc', num_hidden=num_outputs)
new_symbol = mx.sym.SoftmaxOutput(data=new_symbol, name='softmax')
new_symbol = mx.symbol.Flatten(data=new_symbol, name='kp_flat')

# see whats inside this symbolic net
ns_args = new_symbol.list_arguments()
ns_attr = new_symbol.attr_dict()
ns_out = new_symbol.get_internals()
ns_out1 = ns_out.list_outputs()
ns_is = ns_out.infer_shape(data=(2,10))
ns_str = ns_out.debug_str()

al_args = all_layers.list_arguments()
al_attr = all_layers.attr_dict()

#bind the symbol program to a module for execution
# mod = mx.mod.Module(symbol=new_symbol, context=mx.gpu(), label_names=('KP_softmax_label',))
mod = mx.mod.Module(symbol=new_symbol, context=mx.gpu())

#offer up some training data
mod.bind(data_shapes=itr_train.provide_data, label_shapes=itr_train.provide_label)
mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))

mod.fit(train_data=itr_train, eval_data = itr_val,eval_metric='acc', batch_end_callback =mx.callback.Speedometer(batch_size=es.batch_size,frequent=10), epoch_end_callback=es.epoc_end_callback_kp,optimizer='sgd',optimizer_params={'learning_rate':0.1,'momentum': 0.9},num_epoch=es.num_epoch)

pass