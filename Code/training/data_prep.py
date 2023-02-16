import cv2
import numpy as np
import os



def load_images(base_path):
    # training images
    Classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    images = []
    for category in Classes:
        path = os.path.join(base_path, category)
        class_num = Classes.index(category)
        for name in os.listdir(path):
            img = cv2.imread(path + "/" + str(name), 0)
            if img is not None:
                images.append([img, class_num])
    return images

# read images from folder
def load_images_folder():
    # training images
    images_train = load_images('D:/PycharmProject/rfea/dataset/train/')
    # validation images
    images_cv = load_images('D:/PycharmProject/rfea/dataset/valid/')
    # test images
    images_test = load_images('D:/PycharmProject/rfea/dataset/test/')
    
    return images_train, images_cv, images_test
    

# load the images
images_train, images_cv, images_test = load_images_folder()

# change to numpy matrix
images_train = np.array(images_train, dtype=object)
images_cv = np.array(images_cv, dtype=object)
images_test = np.array(images_test, dtype=object)

X_train = []
y_train = []
for features, label in images_train:
    X_train.append(features)
    y_train.append(label)

X_val = []
y_val = []
for features, label in images_cv:
    X_val.append(features)
    y_val.append(label)

X_test = []
y_test = []
for features, label in images_test:
    X_test.append(features)
    y_test.append(label)

# save the numpy matrix
np.save('D:/PycharmProject/rfea/dataset/train_raw.npy', X_train)
np.save('D:/PycharmProject/rfea/dataset/cv_raw.npy', X_val)
np.save('D:/PycharmProject/rfea/dataset/test_raw.npy', X_test)

np.savetxt("D:/PycharmProject/rfea/dataset/y_train.csv", y_train, delimiter=",", fmt='%s')
np.savetxt("D:/PycharmProject/rfea/dataset/y_val.csv", y_val, delimiter=",", fmt='%s')
np.savetxt("D:/PycharmProject/rfea/dataset/y_test.csv", y_test, delimiter=",", fmt='%s')