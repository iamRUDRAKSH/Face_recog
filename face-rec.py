import face_recognition
import cv2
import numpy
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

rud_image = face_recognition.load_image_file("D:\python codes\Projects\Facerecognition\Faces\rudraksh.jpg")
rud_encoding = face_recognition.face_encodings(rud_image)[0]
kinchu_image = face_recognition.load_image_file("D:\python codes\Projects\Facerecognition\Faces\kanchan.jpg")
kinchu_encoding = face_recognition.face_encodings(kinchu_image)[0]
manya_image = face_recognition.load_image_file("D:\python codes\Projects\Facerecognition\Faces\manaswi.jpg")
manya_encoding = face_recognition.face_encodings(manya_image)[0]

known_face_encodings = [rud_image, kinchu_encoding, manya_encoding]
known_face_names = ["Rudraksh", "Kanchan", "Manaswi"]

# Initialize some variables
People = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")
f = open(f"{current_date}.csv", "w+", newline="")
writer = csv.writer(f)



while True:
    # Grab a single frame of video
    _ , frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distanes(known_face_encodings, face_encoding)
        best_match_index = numpy.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name in known_face_names:
            font = cv2.FONT_TIMES_NEW_ROMAN
            pos = (10, 100)
            fontscale = 1.5
            color = (0, 0, 255)
            thickness = 2
            linetype = 2
            cv2.putText(frame, name, pos, font, fontscale, color, thickness, linetype)
            
            if name in People:
                People.remove(name)
                current_time = now.strftime("%H:%M:%S")
                writer.writerow([name, current_time])



    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()
      
