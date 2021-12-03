import cv2

# load pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('Xmen.jpg.')  # choosing file(here image to detect faces)
video = cv2.VideoCapture('The Social Network 2010 best scene - Justine Rd_Trim2.mp4')

while True:

    successful_frame_read, frame = video.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Face detector', frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
video.release()

# grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting img to greyscale

"""

# print(face_coordinates)
# draw rectangle around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# print(face_coordinates)
cv2.imshow('Face detected', frame)
cv2.waitKey()
"""
