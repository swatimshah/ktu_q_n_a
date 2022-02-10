from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import tensorflow
from numpy import mean
from sklearn.preprocessing import RobustScaler

rScaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(20, 100-20), unit_variance=True)

# load the test data
X = loadtxt('d:\\eeg\\test_input_one_640.csv', delimiter=',')

# normalize the data 
input = rScaler.fit_transform(X[:, 0:640])
savetxt('d:\\input-swati-online.csv', input, delimiter=',')

# transform the data in the format which the model wants
input = input.reshape(len(input), 10, 64)
input = input.transpose(0, 2, 1)

# get the expected outcome 
y_real = X[:, -1]

# load the model
model = load_model('D:\\eeg\\model_conv1d.h5')

# get the "predicted class" outcome
y_pred = model.predict_proba(input) 
print(y_pred.shape)
y_max = numpy.argmax(y_pred, axis=1)

# calculate the confusion matrix
matrix = confusion_matrix(y_real, y_max)
print(matrix)
