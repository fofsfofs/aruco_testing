import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)
parameters = aruco.DetectorParameters_create()
markerDictionary = aruco.Dictionary_get(aruco.DICT_6X6_1000)

while True:
    # Capture frame-by-frame
    boolAndFrame = cap.read()
    ret = boolAndFrame[0]
    frame = boolAndFrame[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, markerDictionary, parameters=parameters)
    aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow('Detecting markers', cv2.flip(frame, 1))

    # if ids is not None:
    #   print(ids)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
