import numpy
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from matplotlib import pyplot
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
from numpy import mean
from tensorflow.random import set_seed
import tensorflow
from keras.constraints import min_max_norm
from keras.regularizers import L2
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import RobustScaler

rScaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(20, 100-20), unit_variance=True)

# setting the seed
seed(1)
set_seed(1)

# load array
X_train_whole = loadtxt('d:\\eeg\\combined_eeg_handle_640.csv', delimiter=',')

#shuffle the data
numpy.random.shuffle(X_train_whole)

# split the data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(X_train_whole, X_train_whole[:, -1], random_state=1, test_size=0.3, shuffle = True)
print(X_train_tmp.shape)
print(X_test_tmp.shape)

# augment train data
choice = X_train_tmp[:, -1] == 0.
X_total_1 = numpy.append(X_train_tmp, X_train_tmp[choice, :], axis=0)
X_total_2 = numpy.append(X_total_1, X_train_tmp[choice, :], axis=0)
X_total_3 = numpy.append(X_total_2, X_train_tmp[choice, :], axis=0)
X_total_4 = numpy.append(X_total_3, X_train_tmp[choice, :], axis=0)
X_total = numpy.append(X_total_4, X_train_tmp[choice, :], axis=0)


print(X_total.shape)

# data balancing for train data
sm = SMOTE(random_state = 2)
X_train_keep, Y_train_keep = sm.fit_resample(X_total[:, 0:640], X_total[:, -1].ravel())
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_keep == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_keep == 0)))

#combine the input and the label of the train data
train_combined = numpy.append(X_train_keep, Y_train_keep.reshape(len(Y_train_keep), 1), axis=1) 
numpy.random.shuffle(train_combined)
X_train = train_combined[:, 0:640]
Y_train = train_combined[:, -1]

#combine the input and the label of the validation data
test_combined = numpy.append(X_test_tmp, Y_test_tmp.reshape(len(Y_test_tmp), 1), axis=1) 
numpy.random.shuffle(test_combined)
X_test = test_combined[:, 0:640]
Y_test = test_combined[:, -1]



#=======================================
 
# Data Pre-processing

# normalize the training data
input = rScaler.fit_transform(X_train)
input_output = numpy.append(input, Y_train.reshape(len(Y_train), 1), axis=1) 
savetxt('d:\\input_output.csv', input_output, delimiter=',')

# normalize the validation data
testinput = rScaler.fit_transform(X_test)
savetxt('d:\\testinput.csv', testinput, delimiter=',')


#=====================================

# Model configuration

print(len(input))
print(len(testinput))

input = input.reshape(len(input), 10, 64)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 10, 64)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)



# Create the model
model=Sequential()
model.add(Conv1D(filters=20, kernel_size=6, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), kernel_constraint=min_max_norm(min_value=-0.1, max_value=0.1), padding='valid', activation='relu', strides=3, input_shape=(64, 10)))
model.add(BatchNormalization())
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(filters=30, kernel_size=4, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), kernel_constraint=min_max_norm(min_value=-0.1, max_value=0.1), padding='valid', activation='relu', strides=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(2, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), kernel_constraint=min_max_norm(min_value=-0.1, max_value=0.1), activation='softmax'))

model.summary()

# 
#model.add(Conv1D(filters=30, kernel_size=4, kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005), kernel_constraint=min_max_norm(min_value=-0.05, max_value=0.05), padding='valid', #activation='relu', strides=2))
#model.add(Dropout(0.4))
#model.add(BatchNormalization())
#model.add(Conv1D(filters=20, kernel_size=4, kernel_regularizer=L2(0.00005), bias_regularizer=L2(0.00005), kernel_constraint=min_max_norm(min_value=-0.05, max_value=0.05), padding='valid', #activation='relu', strides=1))
#model.add(Conv1D(filters=20, kernel_size=6, kernel_regularizer=L2(0.0001), bias_regularizer=L2(0.0001), kernel_constraint=min_max_norm(min_value=-1, max_value=1), padding='valid', #activation='relu', strides=1))
#, data_format='channels_first'

# Compile the model
adam = Adam(learning_rate=0.004, epsilon=1)       
model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

hist = model.fit(input, Y_train, batch_size=24, epochs=800, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None)


# evaluate the model
Y_hat_classes = model.predict_classes(testinput)
matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training and validation history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.xlabel("No of iterations")
pyplot.ylabel("Accuracy and loss")
pyplot.show()

#==================================

model.save("D:\\eeg\\model_conv1d.h5")

#==================================

#Removed dropout and reduced momentum and reduced learning rate