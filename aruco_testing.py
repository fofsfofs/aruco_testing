import cv2
import cv2.aruco as aruco

# Initializing capture device
cap = cv2.VideoCapture(0)
parameters = aruco.DetectorParameters_create()
markerDictionary = aruco.Dictionary_get(aruco.DICT_6X6_1000)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    # Capture frame-by-frame
    boolAndFrame = cap.read()
    ret = boolAndFrame[0]
    frame = boolAndFrame[1]
    # Retrieving grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 40)

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, 0.1 * perimeter, True)

        if len(poly) == 4:
            x, y, w, h = cv2.boundingRect(c)
            location = ["top/bottom", "right/left"]
            if cv2.contourArea(c) > 5625:
                if y < cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2:
                    # print(str(x), end=" ")
                    location[0] = "TOP"
                    print("top", end=" ")
                else:
                    # print(str(x), end=" ")
                    location[0] = "BOTTOM"
                    print("bottom", end=" ")

                if x > cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2:
                    # print(str(y))
                    location[1] = " RIGHT"
                    print("right")
                else:
                    # print(str(y))
                    location[1] = " LEFT"
                    print("left")
                cv2.putText(frame, location[0] + location[1], (x, y), font, 1, (0, 0, 0))

    corners, ids, _ = aruco.detectMarkers(gray, markerDictionary, parameters=parameters)
    # Actual detection of the markers
    aruco.drawDetectedMarkers(frame, corners, ids)

    # Showing the detected markers along with flipping the view along the y-axis

    # cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
    cv2.imshow('Detecting markers', frame)
    cv2.imshow('Threshold', cv2.flip(threshold, 1))

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
