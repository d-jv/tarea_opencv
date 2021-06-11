import cv2
import sys

# With values inside code
# imagePath = "genteDeTalentoDigital.jpg"
# cascPath = "haarcascade_frontalface_default.xml"

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale( #The detectMultiScale function is a general function that detects objects. Since we are calling it on the face cascade, that’s what it detects
    gray,
    scaleFactor=1.1, # Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this.
    minNeighbors=5, # defines how many objects are detected near the current one before it declares the face found. 
    minSize=(30, 30), # gives the size of each window.
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE    Now its legacy code, no longer allows importing the cv module.
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)