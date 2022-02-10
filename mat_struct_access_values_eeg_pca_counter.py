import scipy.io as sio
import numpy
from numpy import savetxt 
from sklearn.decomposition import PCA

def _check_keys( dict):
	"""
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	"""
	for key in dict:
    		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
        		dict[key] = _todict(dict[key])
	return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
	"""
	this function should be called instead of direct scipy.io .loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	"""
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return _check_keys(data)

combinedData = numpy.empty([0, 640])
labelOne = numpy.ones((221, 1))

for i in range (221):
	myKeys = loadmat("D:\\eeg\\z_one_test\\filtered\\filtered\\Yes\\eeg" + str(i+1) + ".mat")
	print(myKeys)
	eegData = myKeys['eeg_handle']
	pca_target = PCA(n_components=10)
	pca_target.fit(eegData)
	print(pca_target.components_.shape)
	combinedData = numpy.append(combinedData, pca_target.components_.flatten().reshape(1, 640), axis=0)

labelZero = numpy.zeros((221, 1))

for i in range (221):
	myKeys = loadmat("D:\\eeg\\z_one_test\\filtered\\filtered\\No\\eeg" + str(i+1) + ".mat")
	print(myKeys)
	eegData = myKeys['eeg_handle']
	pca_target = PCA(n_components=10)
	pca_target.fit(eegData)
	print(pca_target.components_.shape)
	combinedData = numpy.append(combinedData, pca_target.components_.flatten().reshape(1, 640), axis=0)


labels = numpy.append(labelOne, labelZero, axis=0)

combinedData = numpy.append(combinedData, labels, axis=1)
savetxt('d:\\eeg\\z_one_test\\test_input_one_640.csv', combinedData, delimiter=',')


