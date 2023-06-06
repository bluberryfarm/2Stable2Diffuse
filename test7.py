import os 



if __name__ == "__main__":
    
    # assign directory
    directory = os.getcwd()+"/Rock_Photos/Real_Rock/"
    

    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
    
            """
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