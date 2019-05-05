import cv2
import numpy as np

#
def makecords(image, line_parameters):
    slope, intercept = line_parameters
    print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

#smoothens out lines and averages them
def average_slope_intercept(image, lines):
    leftfit = []
    rightfit = []
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            leftfit.append((slope, intercept))
        else:
            rightfit.append((slope, intercept))
    lfavg = np.average(leftfit, axis=0)
    rfavg = np.average(rightfit, axis=0)
    print(lfavg, "hi",  rfavg)
    lfline = makecords(image, lfavg)
    rfline = makecords(image, rfavg)
    return np.array([lfline,rfline])

# uses canny for edge detection (turns image to array than detects big changes in color)
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

#creates lines on detected lanes
def display_lines(image, lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image

#crops out edges outside of the front of the car
def region_of_intrest(image):
    height=image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# This section is for image lane detection:

# image = cv2.imread("test_image.jpg")
# laneimage = np.copy(image)
# cannyimage = canny(laneimage)
# cropped_image = region_of_intrest(cannyimage)
# lines = cv2.HoughLinesP(cropped_image,2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5 )
# avglines=average_slope_intercept(laneimage, lines)
# line_image = display_lines(laneimage, avglines)
# combo_image = cv2.addWeighted(laneimage, 0.8, line_image, 1, 1)
# cv2.imshow("results", combo_image)
# cv2.waitKey(0)

# this section is for video line detection:
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):

    #seperates video to frames and runs canny+crops region_of_intrest
    _, frame = cap.read()
    cannyimage = canny(frame)
    cropped_image = region_of_intrest(cannyimage)

    #detects lines with HoughLinesP and smoothens+displays them
    lines = cv2.HoughLinesP(cropped_image,2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5 )
    avglines=average_slope_intercept(frame, lines)
    line_image = display_lines(frame, avglines)

    #overlays image with frame and shows results
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("results", combo_image)

    #allows to exit video with "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
