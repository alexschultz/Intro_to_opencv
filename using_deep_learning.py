import os
import numpy as np
import cv2


def show_faces(net, cfd=.60):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > cfd:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(img, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



        cv2.imshow('webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(os.path.join('model', 'deploy.prototxt.txt'),
                                   os.path.join('model', 'res10_300x300_ssd_iter_140000.caffemodel'))
    show_faces(net)


if __name__ == '__main__':
    main()
