import cv2

trained_faces = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
img = cv2.imread(input("enter an img name with extension: "))

grayscaled_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_faces.detectMultiScale(grayscaled_img)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

cv2.imshow("face detector", img)
cv2.waitKey()
