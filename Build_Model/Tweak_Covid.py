# # Using MobileNet for our Covid Detection
# 
# ### Loading the MobileNet Model

# Freeze all layers except the top 4, as we'll only be training the top 4

from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# MobileNet was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224 

# Re-loads the MobileNet model without the top or FC layers
MobileNet = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers 
# Layers are set to trainable as True by default
for layer in MobileNet.layers:
    layer.trainable = False
    
# Let's print our layers 
# for (i,layer) in enumerate(MobileNet.layers):
#    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# ### Let's make a function that returns our FC Head

def build(bottom_model, num_classes, neurons):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(neurons,activation='relu')(top_model)
    top_model = Dense(neurons,activation='relu')(top_model)
    top_model = Dense(neurons//2,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    file1 = open("tweak_accuracy.txt", "a")
    print("No. of Layers:", 28)
    file1.write("No. of Layers: 28"+"\n")
    print("Number of layer trained: ", 5)
    file1.write("Number of layer trained: 5"+"\n")
    print("No. of neuron in top trainable layer: ", neurons)
    file1.write("No. of neuron in top trainable layer: " + str(neurons)+"\n")
    file1.close()
    return top_model


def resetWeight(model):
	print("Reseting weights")
	w = model.get_weights()
	w = [[j*0 for j in i] for i in w]
	model.set_weights(w)


# ### Let's add our FC Head back onto MobileNet

num_classes = 2   # covid and normal

FC_Head = build(MobileNet, num_classes, 1024)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

with open('tweak_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

# ### Loading our Covid Dataset

train_data_dir = 'dataset/train/'
validation_data_dir = 'dataset/validation/'

# Enter the number of training and validation samples here
nb_train_samples = 40
nb_validation_samples = 10

# We only train 5 EPOCHS 
epochs = 5
batch_size = 16
count = 0


def postProcessing():
	# Let's use some data augmentaiton 
	train_datagen = ImageDataGenerator(
	      rescale=1./255,
	      rotation_range=45,
	      width_shift_range=0.3,
	      height_shift_range=0.3,
	      horizontal_flip=True,
	      fill_mode='nearest')
	 
	validation_datagen = ImageDataGenerator(rescale=1./255)
	  
	train_generator = train_datagen.flow_from_directory(
	        train_data_dir,
	        target_size=(img_rows, img_cols),
	        batch_size=32,
	        class_mode='categorical')
	 
	validation_generator = validation_datagen.flow_from_directory(
	        validation_data_dir,
	        target_size=(img_rows, img_cols),
	        batch_size=batch_size,
	        class_mode='categorical')


	# We use a very small learning rate 
	model.compile(loss = 'categorical_crossentropy',
	              optimizer = Adam(lr = 0.001),
	              metrics = ['accuracy'])



	history = model.fit_generator(
	    train_generator,
	    steps_per_epoch = nb_train_samples // batch_size,
	    epochs = epochs,
	    verbose = 0,
	    validation_data = validation_generator,
	    validation_steps = nb_validation_samples // batch_size)

	accuracy = max(history.history['accuracy'])

	print("Accuracy: ",accuracy)

	return accuracy

accuracy = postProcessing()
file1 = open("tweak_accuracy.txt", "w")
file1.write(str(accuracy*100)+"\n")
file1.close()

best_acc = accuracy
best_neu = 1024

while accuracy < 0.95 and count < 4:
	file2 = open("tweak_accuracy.txt", "a")
	print("\t\tAttempt ",count+1)
	file2.write("\t\tAttempt " + str(count+1) + "\n")
	neuron = 100*(count+1)*3
	FC_Head1 = build(MobileNet, num_classes, neuron)
	model = Model(inputs = MobileNet.input, outputs = FC_Head)
	accuracy = postProcessing()
	if accuracy > best_acc:
		best_acc = accuracy
		best_neu = neuron
	
	file2.write(str(accuracy*100) + "\n")
	file2.close()
	resetWeight(model)
	count = count + 1

print("\nBest Accuracy: ", best_acc)

model.save('covid_model.h5')
print("Best model saved!!!")
