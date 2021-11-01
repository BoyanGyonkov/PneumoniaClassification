import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
import tensorflow_addons as tfa 
import numpy as np
import cnn_t
import matplotlib.pyplot as plt
import random

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    #read as grayscale
    image= tf.image.decode_jpeg(image_string, channels=1)

    image= tf.image.convert_image_dtype(image, tf.float32)

    #crop the center of the image, where the lungs are located
    image = tf.image.central_crop(image, 0.92)
    image= tf.image.resize(image, size=[256,256])

    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    #rotate between 30 and 18 degrees in a random direction
    direc = 1 if random.random() <0.5 else -1
    deg = random.randint(6,10) 
    image = tfa.image.rotate(image, direc*tf.constant(np.pi/deg), fill_mode = 'nearest')
    #shift up to 10 pixels to the left or right
    image = tfa.image.translate(image, [random.randint(0,20)-10 , random.randint(0,20) - 10] , fill_mode = 'nearest')

    return image, label

#get paths of training images
healthy_im_paths_train = [os.path.join("C:/chest_xray/train/NORMAL",f) for f in os.listdir("C:/chest_xray/train/NORMAL") if os.path.isfile(os.path.join("C:/chest_xray/train/NORMAL", f))]
pnem_im_paths_train = [os.path.join("C:/chest_xray/train/PNEUMONIA",f) for f in os.listdir("C:/chest_xray/train/PNEUMONIA") if os.path.isfile(os.path.join("C:/chest_xray/train/PNEUMONIA", f))]
im_paths_train = healthy_im_paths_train + pnem_im_paths_train

#generate the labels for the images
labels1_train = np.zeros(len(healthy_im_paths_train))
labels2_train = np.ones(len(pnem_im_paths_train))
labels_train = np.concatenate((labels1_train, labels2_train), axis=0)

#create dataset as pair: (image, label)
dataset_health_train = tf.data.Dataset.from_tensor_slices((healthy_im_paths_train, labels1_train))
dataset_health_train = dataset_health_train.map(parse_function, num_parallel_calls = 4)

#Use data augmentation on healthy patients only, to soften the effects of the class imbalance (Normal: 1300 images , Pneumonia: 3900 iamges)
dataset_health_train1 = dataset_health_train.map(augment, num_parallel_calls = 4)
dataset_health_train2 = dataset_health_train.map(augment, num_parallel_calls = 4)
dataset_health_train = (dataset_health_train.concatenate(dataset_health_train1)).concatenate(dataset_health_train2)


dataset_pn_train = tf.data.Dataset.from_tensor_slices((pnem_im_paths_train, labels2_train))
dataset_pn_train = dataset_pn_train.map(parse_function, num_parallel_calls =4)


dataset_train = dataset_health_train.concatenate(dataset_pn_train)
dataset_train = dataset_train.shuffle(20000)
dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)

#repeat procedure for test images
healthy_im_paths_val = [os.path.join("C:/chest_xray/test/NORMAL",f) for f in os.listdir("C:/chest_xray/test/NORMAL") if os.path.isfile(os.path.join("C:/chest_xray/test/NORMAL", f))]
pnem_im_paths_val = [os.path.join("C:/chest_xray/test/PNEUMONIA",f) for f in os.listdir("C:/chest_xray/test/PNEUMONIA") if os.path.isfile(os.path.join("C:/chest_xray/test/PNEUMONIA", f))]
im_paths_val = healthy_im_paths_val + pnem_im_paths_val

labels1_val = np.zeros(len(healthy_im_paths_val))
labels2_val = np.ones(len(pnem_im_paths_val))
labels_val = np.concatenate((labels1_val, labels2_val), axis=0)

dataset_val = tf.data.Dataset.from_tensor_slices((im_paths_val , labels_val))
dataset_val = dataset_val.map(parse_function, num_parallel_calls = 4)
dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)


model = cnn_t.SequentialModel(out_function = 'sigmoid')
model.add(cnn_t.Conv2D(kernel_size=9 ,in_channels=1, out_channels=32))
model.add(cnn_t.BatchNormalization(32, for_conv = True))
model.add(cnn_t.MaxPool2D(2))

model.add(cnn_t.Conv2D(kernel_size=7 ,in_channels=32, out_channels=64))
model.add(cnn_t.BatchNormalization(64,for_conv = True))
model.add(cnn_t.MaxPool2D(2, padding = "SAME"))

model.add(cnn_t.Conv2D(kernel_size=5 ,in_channels=64, out_channels=128))
model.add(cnn_t.BatchNormalization(128,for_conv = True))
model.add(cnn_t.MaxPool2D(2, padding = "SAME"))

model.add(cnn_t.Flatten())

model.add(cnn_t.Dense(200))
model.add(cnn_t.BatchNormalization(200))
model.add(cnn_t.Dropout(0.2))
model.add(cnn_t.Dense(200))
model.add(cnn_t.BatchNormalization(200))
model.add(cnn_t.Dropout(0.2))
model.add(cnn_t.Dense(1))
model.build()
model.printLayers()

epochs = 10


evalHistory = model.fit(dataset_train , epochs, 0.001, dataset_val, 30)

#imPath = "C:/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
#image  = parse_function(imPath,1)[0]
#print(model.prediction(tf.reshape(image, shape=(1, image.shape[0], image.shape[1], image.shape[2])) ))


accuracy = [x[1] for x in evalHistory]
plt.plot(accuracy)
plt.yticks(np.arange(min(accuracy), max(accuracy)+0.1, 0.1))
plt.show()

