"""
Required installed packages:
* Tensorflow
* Numpy
* Matplotlib
* PIL
"""
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os 
import random

from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Code for this model was built in part using a tutorial from Tensorflow. That tutorial can be found below:
# https://www.tensorflow.org/tutorials/images/classification




if __name__ == "__main__":
    # set the seed for tensorflow and python, respectively
    tf.random.set_seed(18000000000000000)
    random.seed(180000000000000000)

    # dataset notes:
    #   * keep data_dir in the same directory as this code, otherwise it will not run
    data_dir = os.getcwd()+"/Rock_Photos/"

    # define the size of your images and the batch size
    batch_size = 32
    img_height = 180
    img_width = 180


    # training dataset
    train_ds = utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # validation dataset
    val_ds = utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    # print(class_names)
    
    # to see the data, uncomment the code below
    # it will generate 9 images of the different data, each labeled with its respective label
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break
    


    AUTOTUNE = tf.data.AUTOTUNE

    # "keeps the images in memory after they're loaded off disk during the first epoch. 
    # This will ensure the dataset does not become a bottleneck while training your model. 
    # If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache."
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    # "overlaps data preprocessing and model execution while training."
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Rescales values to be on the range [0,1] rather than RGB values ranging from 0 to 255
    normalization_layer = layers.Rescaling(1./255)
    # Rescaled using Dataset.map() function
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    # create the model and define the layers of the neural net
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # compile the model using adam optimizer
    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # print model stats/summary to console 
    model.summary()

    # train the model; define epochs
    epochs=10
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

    # get model accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # get model loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    # plot the model accuracy and the model loss using matplotlib.pyplot
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1, xlabel = "Number of Epochs", ylabel = "Accuracy")
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy vs Number of Epochs')

    plt.subplot(1, 2, 2, xlabel = "Number of Epochs", ylabel = "Loss")
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss vs Number of Epochs')
    plt.show()


    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    #    Note: MAKE SURE h5py PACKAGE IS INSTALLED
    # model.save('sumothermodel_SD_Model_HDF5.h5')



    # make predictions on new images using the following code 

    # example if path is stored online
    """
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    """

    image_path = os.getcwd()+"/Rock_Photos/Real_Rock/dwayne-rock-johnson.jpeg"

    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    # uncomment this code to make predictions for each image in the dataset 
    # and print the results to terminal

    """
    # it is a bit excessive so fair warning


    # assign directory
    directory = os.getcwd()+"/Rock_Photos/Fake_Rock/"

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
    
            image_path = f

            img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
            )

    """
    """
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
           print(f)
        
    image_path = os.getcwd()+"/Rock_Photos/Real_Rock/dwayne-rock-johnson.jpeg"

    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    """
