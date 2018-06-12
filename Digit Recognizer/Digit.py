import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D as Conv2D
from keras.utils import to_categorical

num_classes = 10
batch_size = 32
epochs = 10

# load data
x_train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = x_train.iloc[:,0]
x_train = x_train.drop(x_train.columns[0],axis=1)

#reshape data
x_train= x_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)


# normalize data
# cross validation
X1,X2,Y1,Y2 = train_test_split(x_train,y_train,test_size=0.3,random_state=0)
#one-hot encoding theLabel
Y1=to_categorical(Y1, num_classes = 10)
Y2=to_categorical(Y2, num_classes = 10)

#print(x_train.shape)
model= Sequential()
model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes)) #output :10 dim
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
model.compile(loss='categorical_crossentropy',
			  optimizer = opt,
			  metrics =['accuracy'])

datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=False,  # randomly flip images
		vertical_flip=False)  # randomly flip images

datagen.fit(X1)

model.fit_generator(datagen.flow(X1, Y1,
                        batch_size=batch_size),
                        epochs=epochs,
                        steps_per_epoch=X1.shape[0],
                        validation_data=(X2, Y2))
scores = model.evaluate(X2, Y2, verbose=1)

Y_test = model.predict(test,batch_size=batch_size)

with open(result.csv,'w') as f:
	f.write('imageId,Label\n')
	for i in range(len(Y_test)):
		f.write("".join([str(i+1),',',str(Y_test[i]),'\n']))