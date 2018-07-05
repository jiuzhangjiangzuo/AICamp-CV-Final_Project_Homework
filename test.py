from detect.MtcnnDetector import MtcnnDetector
from detect.detector import Detector
from detect.fcn_detector import FcnDetector
from detect.mtcnn_model import P_Net, R_Net, O_Net
from recognize.facenet import FaceNet
import cv2
import numpy as np

# Init MtcnnDetector
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = []
prefix = ['detect/MTCNN_model/PNet_landmark/PNet', 'detect/MTCNN_model/RNet_landmark/RNet', 'detect/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

detectors.append(FcnDetector(P_Net, model_path[0]))
detectors.append(Detector(R_Net, 24, batch_size[1], model_path[1]))
detectors.append(Detector(O_Net, 48, batch_size[2], model_path[2]))

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

# Init FaceNet
model_path = 'recognize/facenet_model'
face_net = FaceNet(model_path)

img1 = cv2.imread('/Users/Zhou/Documents/SARI/pictures/20151026223216.jpg')
img2 = cv2.imread('/Users/Zhou/Documents/SARI/pictures/20151026144342.jpg')

faces1 = mtcnn_detector.get_face_from_single_image(img1)

faces2 = mtcnn_detector.get_face_from_single_image(img2)

print(len(faces1))
print(len(faces2))

emb1 = face_net.predict(faces1[0])
emb2 = face_net.predict(faces2[0])

print("score: ")
print(np.sqrt(np.sum(np.square(np.subtract(emb1, emb2)))))