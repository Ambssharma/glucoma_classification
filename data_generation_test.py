from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.1,
        zoom_range=0.2,
        vertical_flip=True,
        fill_mode='constant',zca_whitening=True, zca_epsilon=1e-5,featurewise_center=True)

import glob
cv_img = []
for img in glob.glob("train/non-glaucoma resized/*.jpg"):
    # n= cv2.imread(img)
    # cv_img.append(n)

    img = load_img(img)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='train/final_non_glu_aug', save_prefix='new', save_format='jpg'):
        i += 1
        if i > 0:
            break  # otherwise the generator would loop indefinitely