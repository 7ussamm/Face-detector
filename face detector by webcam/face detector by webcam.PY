import cv2

trained_faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

while True:
    
    frame_0, frame = webcam.read()

    grayscaled_img =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    face_coordinates = trained_faces.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow("face detector", frame)
    key = cv2.waitKey(1)

    if key== 27:
        break

webcam.release()