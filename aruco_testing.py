import cv2
import cv2.aruco as aruco

# Initializing capture device
cap = cv2.VideoCapture(0)
parameters = aruco.DetectorParameters_create()
markerDictionary = aruco.Dictionary_get(aruco.DICT_6X6_1000)

while True:
    # Capture frame-by-frame
    boolAndFrame = cap.read()
    ret = boolAndFrame[0]
    frame = boolAndFrame[1]
    # Retrieving grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, markerDictionary, parameters=parameters)
    # Actual detection of the markers
    aruco.drawDetectedMarkers(frame, corners, ids)

    # Showing the detected markers along with flipping the view along the y-axis
    cv2.imshow('Detecting markers', cv2.flip(frame, 1))

    # if ids is not None:
    #   print(ids)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
