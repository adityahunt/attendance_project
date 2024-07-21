import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            labels.append(os.path.splitext(filename)[0])
    return images, labels

def encode_faces(images):
    face_encodings = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face_encodings.append(gray[y:y+h, x:x+w])
    return face_encodings

def recognize_face(face_encodings, test_image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        test_face = gray[y:y+h, x:x+w]
        for i, face_encoding in enumerate(face_encodings):
            if np.array_equal(test_face, face_encoding):
                return i
    return -1
