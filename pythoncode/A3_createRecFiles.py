"""
This file is used to generate the lst files for test/tran/val and then
it uses these list files to generate the .rec files

python im2rec.py -h #for help
"""

import subprocess
import os
import settings
import errno

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise #
def makeLstFile(prefix, images_loc):
    """
    :param outputfile: name of the list file
    :param imageloc: where images are located, both positive and negative samples
    :return:
    """
    #the recursive means to walk through subfolders and give labels based on which folder image is in
    #NOTE im2rec does not accept .png files, in im2rec.py, change the following
    #cgroup.add_argument('--exts', type=list, default=['.jpeg', '.jpg']
    #to
    #cgroup.add_argument('--exts', type=list, default=['.jpeg', '.jpg', '.png']
    print subprocess.check_output("python ~/mxnet/tools/im2rec.py --list True --recursive True " + prefix + " " + images_loc, shell=True)
    silentremove(prefix+"_test.lst")
    silentremove(prefix+"_val.lst")

    #os.rename(os.path.join(prefix + "_train.lst"), os.path.join(root, prefix + ".lst"))

def makeRecFile(prefix, imageloc):
    """
    :param prefix: name of the list file
    :param imageloc: where images are located, both positive and negative samples
    :return:
    """
    subprocess.call("python ~/mxnet/tools/im2rec.py --encoding .png " + prefix + " " + imageloc, shell=True)

#first create the directory
settings.makeDir(settings.record_IO_directory)

#get the absolute path
root = os.path.abspath(os.path.dirname(__file__))
recordIODir = os.path.join(root, settings.record_IO_directory)

#some locations
val = os.path.join(recordIODir,"Val")
test = os.path.join(recordIODir,"Test")
train = os.path.join(recordIODir,"Train")
valdir = os.path.join(root,settings.mxnet_images_val_dir)
testdir = os.path.join(root,settings.mxnet_images_test_dir)
traindir = os.path.join(root,settings.mxnet_images_train_dir)

#first lets create the list files
makeLstFile(val, valdir)
makeLstFile(test, testdir)
makeLstFile(train, traindir)

#now the rec files
makeRecFile(val, valdir)
makeRecFile(test, testdir)
makeRecFile(train, traindir)