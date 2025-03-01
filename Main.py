import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from cv2 import FONT_HERSHEY_COMPLEX

# Initialize the webcam
cap = cv2.VideoCapture(1)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Load the Encoding file
print("Loading Encode File...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)

encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded...")

while True:
    success, img = cap.read()
    if not success:
        continue

    # Resize frame for faster processing
    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings in the current frame
    faceCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back to original size
            bbox = (x1, y1, x2 - x1, y2 - y1)
            img = cvzone.cornerRect(img, bbox, rt=0)  # Draw rectangle around the detected face
            cv2.putText(img, "Registered", (x1, y1 - 10), FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Show the full webcam feed without any background image
    cv2.imshow("Face Recognition", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
