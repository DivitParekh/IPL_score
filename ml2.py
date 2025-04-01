import cv2
import face_recognition
import numpy as np
import os

# Load known faces
KNOWN_FACES_DIR = "known_faces"  # Folder with known face images
known_faces = []
known_names = []

# Load images and encode faces
for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:  # Ensure face encoding is found
        known_faces.append(encoding[0])
        known_names.append(os.path.splitext(filename)[0])  # Use filename as name

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations & encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
        name = "Unknown"
        
        if True in matches:
            match_index = np.argmax(matches)
            name = known_names[match_index]
        
        # Draw a rectangle around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    
    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
