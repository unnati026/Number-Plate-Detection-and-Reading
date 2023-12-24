from ultralytics import YOLO
import numpy as np
import cv2
import pytesseract
from PIL import Image
import re


def remove_non_alphanumeric(input_string):
    # Use regex to remove non-alphanumeric characters
    result = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return result


# Load YOLO model
model = YOLO('best.pt')

# Load the original video
video_path = 'vid1.mp4'
cap = cv2.VideoCapture(video_path)

pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR/tesseract.exe'

allowed = ['SLW287R']

text = ''

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, conf=0.35, verbose=False)

    lic_list = results[0].boxes.cls.tolist()

    image = results[0].plot()
    # print(image)

    if len(lic_list) != 0:
        x_min, y_min, x_max, y_max = map(int, results[0].boxes.xyxy[0])

        # Crop the image
        img = Image.fromarray(frame)
        img = img.crop((x_min, y_min, x_max, y_max))

        if img.size == 0:
            print(x_min, y_min, x_max, y_max)

        img = cv2.bilateralFilter(np.array(img), 11, 11, 17)

        # apply image de-noising
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        read = pytesseract.image_to_string(dst)
        text = 'Licence Plate detected, number: ' + remove_non_alphanumeric(read)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1

        size = cv2.getTextSize(text, font, font_scale, thickness)
        x = 0
        y1 = int((2 + size[0][1]))

        # # Calculate the position to center the text
        # x = (image.shape[1] - size[0][0]) // 2
        # y = (image.shape[0] + size[0][1]) // 2

        cv2.putText(image, text, (x, y1), font, font_scale, (0, 255, 0), thickness)

        print(text)

        cv2.imshow('Camera', image)

    if len(text) != 0:
        cv2.putText(image, text, (x, y1), font, font_scale, (0, 255, 0), thickness)
    cv2.imshow('Camera', image)

    key = cv2.waitKey(1)
    if key != -1 and key != 13:
        break

cap.release()
cv2.destroyAllWindows()
