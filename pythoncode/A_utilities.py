

import mxnet as mx

class HighestValidationHandler(object):
    '''
    keeps track of highest validation accuracy
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self._highest_validation = 0.0

    def test(self, new_validation):
        retval = new_validation > self._highest_validation
        if retval == True:
            self._highest_validation = new_validation
        return retval

    @property
    def highest_validation(self):
        return self._highest_validation


r=HighestValidationHandler()
paramFile= "AAAAA"

def reset_epoch_end_callback(dest_file = paramFile):
    global paramFile,r
    paramFile = dest_file
    r.reset()

def epoc_end_callback_kp(epoch, symbol, arg_params, aux_params,epoch_train_eval_metrics):
    '''
    early stopping
    a function passed to fit(..) thats called at end of every epoch
    epoch_train_eval_metrics added to track training and validation accuracies both here and in
    mxnets base_module.py fit function.  If this function returns false, fit will stop training.
    Note: fit has been modified for this to work
    :param epoch:
    :param symbol:
    :param arg_params:
    :param aux_params:
    :param epoch_train_eval_metrics:
    :return:
    '''
    EVAL_ACC_TRAIN = 0
    EVAL_ACC_VAL = 1

    for key in epoch_train_eval_metrics.keys():
        for epoch in epoch_train_eval_metrics[key].keys():
            len1 = len(epoch_train_eval_metrics[key])-1
            val_acc = epoch_train_eval_metrics[key][len1][EVAL_ACC_VAL]
            trn_acc = epoch_train_eval_metrics[key][len1][EVAL_ACC_TRAIN]
            retval =  val_acc<1.0 and trn_acc<1.0

            if r.test(val_acc)== True:
                cb = mx.callback.do_checkpoint(paramFile, period = 1)
                cb(1,symbol,arg_params, aux_params)
                print("Highest accuracy =%f"%r.highest_validation)

            if retval == False:
                print("100 percent accuracy for %s stopping"%("val_acc" if val_acc>= 1.0 else "trn_acc"))
                print("Highest accuracy =%f"%r.highest_validation)

            return retval

def output_diffs(attr_dict_1, attr_dict_2, string_attr_dict1="1", string_attr_dict2= "2",showExcludedKeys=False, showTotalLengths = False):
    '''
    check to see if all keys in attr_dict_1 are in attr_dict_2
    :param attr_dict_1:
    :param attr_dict_2:
    :param string_attr_dict1: what is printed to console
    :param string_attr_dict2: what is printed to console
    :param showExcludedKeys:  if True show all keys in attr_dict_1 that are not in attr_dict_2
    :param showTotalLengths:  if true shows lengths of all dictionaryies
    :return:
    '''
    l1 = len(attr_dict_1)
    l2 = len(attr_dict_2)
    diff = l2 - l1
    if showTotalLengths == True:
        print("size of " +  string_attr_dict1+ "=%d,"%(l1) + " size of " + string_attr_dict2 + "=%d,"%(l2) + " diff = %d" % (diff))

    tot_notin =0
    tot_in = 0
    bprintgreeting = False

    for k in attr_dict_1:
        if k not in attr_dict_2:
            if showExcludedKeys == True:
                if not bprintgreeting:
                    print ("list of keys in " + string_attr_dict1 + "  but not in " + string_attr_dict2)
                    bprintgreeting = True
                print (k)
            tot_notin+=1
        else:
            tot_in+=1

    #is all of 1 in 2?
    if tot_notin == 0:
        print(string_attr_dict1 + " has  %d keys, ALL ARE IN "%(l1) + string_attr_dict2 + "\n\n")
    elif tot_notin > 0:
        print(string_attr_dict1 + " has  %d keys, %d of which are not in "%(l1,tot_notin) + string_attr_dict2 + "\n\n")


def strip_obsolete_params(arg_params,aux_params , new_symbol):
    '''
    arg_params and aux_params contain params from COMPLETE model zoo trained neural net
    new_symbol contains a subset of that model

    this function will remove all keys from arg_params or aux_params that
    do not apply to new_symbol
    IOW if key in arg_params or aux_params and NOT in new_symbol
    then delete that key from arg_params or aux_params

    :param arg_params: original models params
    :param aux_params: original models params
    :param new_symbol: model with new heads params
    :return: new_arg_params, new_aux_params  whittled down to contain only params relevant to new_net_params
    '''
    new_arg_params = dict({k:arg_params[k] for k in arg_params if k in new_symbol.list_arguments()})
    new_aux_params = dict({k: aux_params[k] for k in aux_params if k in new_symbol.attr_dict()})
    return (new_arg_params,new_aux_params )
