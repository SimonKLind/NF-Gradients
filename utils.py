import numpy as np
import cv2 as cv

def get_imagemod(image, size):
    h0, w0 = image.shape[:2]
    ratio = size / max(h0, w0)
    h, w = int(h0 * ratio), int(w0 * ratio)
    dh = (size - h) / 2
    dw = (size - w) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return ratio, w, h, top, left, bottom, right

def process_image_for_yolo(image, size):
    ratio, w, h, top, left, bottom, right = get_imagemod(image, size)

    if ratio != 1.0:
        interp = cv.INTER_AREA if ratio < 1 else cv.INTER_LINEAR
        image = cv.resize(image, (w, h), interpolation=interp)

    image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    image = np.ascontiguousarray(image)

    return image

def load_image_for_yolo(path, size):
    return process_image_for_yolo(cv.imread(path), size)
