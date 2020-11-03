import os
import sys
import cv2
import face_recognition

DISPLAY_IMAGE = True
FONT = cv2.FONT_HERSHEY_PLAIN
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

if "--no-gui" in sys.argv:
    DISPLAY_IMAGE = False

known_faces = []
known_names = []

for face_name in os.listdir("known_faces"):
    if face_name != ".DS_Store":
        for filename in os.listdir(f"known_faces/{face_name}"):
            if filename != ".DS_Store":
                image = face_recognition.load_image_file(f"known_faces/{face_name}/{filename}")
                encoding = face_recognition.face_encodings(image)[0]
                known_faces.append(encoding)
                known_names.append(face_name)

cap = cv2.VideoCapture(0)

while True:
    _, cam_image = cap.read()

    face_locations = face_recognition.face_locations(cam_image)
    face_encodings = face_recognition.face_encodings(cam_image, face_locations)
    for f_enc, f_loc in zip(face_encodings, face_locations):
        results = face_recognition.compare_faces(known_faces, f_enc, 0.5)

        if True in results:
            match = known_names[results.index(True)]

            if DISPLAY_IMAGE:
                top_left = (f_loc[3], f_loc[0])
                bottom_right = (f_loc[1], f_loc[2])
                cv2.rectangle(cam_image, top_left, bottom_right, (0, 255, 0), FRAME_THICKNESS)
                cv2.putText(cam_image, match, (top_left[0], bottom_right[1] + 30), FONT, 2, (0, 255, 0), FONT_THICKNESS)

            print(f"{match} has entered the house")

    if DISPLAY_IMAGE:
        cv2.imshow("Camera Footage", cam_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
