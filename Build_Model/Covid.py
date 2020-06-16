# # Using MobileNet for our Covid Detection
# 
# ### Loading the MobileNet Model

# Freeze all layers except the top 4, as we'll only be training the top 4

from keras.applications import MobileNet

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

def build(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model


# ### Let's add our FC Head back onto MobileNet

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 2   # covid and normal

FC_Head = build(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

with open('summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

# ### Loading our Covid Dataset

from keras.preprocessing.image import ImageDataGenerator

train_data_dir = '/Mini/dataset/train/'
validation_data_dir = '/Mini/dataset/validation/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


# ### Training out Model
# - Note we're using checkpointing and early stopping

from keras.optimizers import Adam

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr = 0.001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 40
nb_validation_samples = 10

# We only train 5 EPOCHS 
epochs = 5
batch_size = 16

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

accuracy = max(history.history['acc'])
model.save('covid_model.h5')

print("Accuracy: ",accuracy)

file1 = open("accuracy.txt", "w")
file1.write("Data Augmentation: " + "True" + "\n")
file1.write("Loss function: " + "categorical_crossentropy" + "\n")
file1.write("Optimizer: " + "Adam" + "\n")
file1.write("Learning Rate: " + str(0.001) + "\n")
file1.write("Epochs: " + str(epochs) + "\n")
file1.write("Batch Size: " + str(batch_size) + "\n")
file1.write("Accuracy: " + str(accuracy*100))
file1.close()

