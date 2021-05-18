from os import environ, chdir
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
chdir(r'C:\Users\wanga\Documents\GitHub\Bird-Cam')
import cv2
import tensorflow as tf
from PIL import Image as PIL
import numpy as np


model = tf.keras.models.load_model(r'OldSavedModels\ExportedModels\model_export')
model.summary()


def preprocess_image(image):
    # return np.array(image.convert('RGB').resize(224, 224))
    return np.array(image.resize((224, 224)))


count = 0
cap = cv2.VideoCapture('part1.avi')
while cap.isOpened():
    ret, frame = cap.read()
    height, width, layers = frame.shape
    new_h = height // 2
    new_w = width // 2
    frame = cv2.resize(frame, (new_w, new_h))
    cv2.imshow('window-name', frame)

    if count % 3 == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = preprocess_image(PIL.fromarray(frame))
        # print(frame.shape, 'THIS IS THE SHAPE!!!!')
        frame = np.reshape(frame, (1, 224, 224, 3))
        # print(model.predict(frame))
        y_pred = np.argmax(model.predict(frame))

        if y_pred == 0:
            print('--BIRD--')
        elif y_pred == 1:
            print('\\\\NOT BIRD//')
        else:
            print('!!SQUIRREL!!')

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    count += 1

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows
