import os
import cv2

cap = cv2.VideoCapture(os.path.join('video', 'deepracer.mp4'))

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # do whatever you want to to frame in here

        # edge detection
        # frame = cv2.Canny(frame, 100, 200)

        # thresholding
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur = cv2.medianBlur(gray, 5)
        # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # frame = thresh.copy()

        # # find contours
        # _, contours, h = cv2.findContours(thresh, 1, 2)
        #
        # for cnt in contours:
        #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        #     print(len(approx))
        #     if len(approx) == 5:
        #         print("pentagon")
        #         cv2.drawContours(frame, [cnt], 0, 255, 2)
        #     elif len(approx) == 3:
        #         print("triangle")
        #         cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
        #     elif len(approx) == 4:
        #         print("square")
        #         cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 2)
        #     elif len(approx) == 9:
        #         print("half-circle")
        #         cv2.drawContours(frame, [cnt], 0, (255, 255, 0), 2)
        #     elif len(approx) > 15:
        #         print("circle")
        #         cv2.drawContours(frame, [cnt], 0, (0, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
