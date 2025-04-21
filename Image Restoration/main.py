import cv2
import numpy as np

img = cv2.imread("noise_img.jpg")
img = cv2.resize(img, (600, 400))

gaussian = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
nl_means = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

methods = {
    0: ("Original", img),
    1: ("Gaussian", gaussian),
    2: ("Median", median),
    3: ("Non-local Means", nl_means)
}

def nothing(x):
    pass

cv2.namedWindow("Denoise GUI")
cv2.createTrackbar("Method", "Denoise GUI", 0, 3, nothing)

while True:
    method_index = cv2.getTrackbarPos("Method", "Denoise GUI")
    label, output = methods[method_index]
    
    display = output.copy()
    cv2.putText(display, f"Method: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Denoise GUI", display)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
