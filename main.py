import base64
import json
import os

import cv2
import questionary
import requests
from requests import RequestException


def preprocess_image(image_file_path, max_width, max_height):
    """Preprocesses input images for AutoML Vision Edge models.

    Args:
        image_file_path: Path to a local image for the prediction request.
        max_width: The max width for preprocessed images. The max width is 640
            (1024) for AutoML Vision Image Classfication (Object Detection)
            models.
        max_height: The max width for preprocessed images. The max height is
            480 (1024) for AutoML Vision Image Classfication (Object
            Detetion) models.
    Returns:
        The preprocessed encoded image bytes.
    """
    # cv2 is used to read, resize and encode images.
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    im = cv2.imread(image_file_path)
    [height, width, _] = im.shape
    if height > max_height or width > max_width:
        ratio = max(height / float(max_width), width / float(max_height))
        new_height = int(height / ratio + 0.5)
        new_width = int(width / ratio + 0.5)
        resized_im = cv2.resize(
            im, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, processed_image = cv2.imencode('.jpg', resized_im, encode_param)
    else:
        _, processed_image = cv2.imencode('.jpg', im, encode_param)
        new_height = height
        new_width = width
        resized_im = im
    return base64.b64encode(processed_image).decode('utf-8'), new_height, new_width, resized_im


def container_predict(image_file_path, image_key, port_number=8501):
    """Sends a prediction request to TFServing docker container REST API.

    Args:
        image_file_path: Path to a local image for the prediction request.
        image_key: Your chosen string key to identify the given image.
        port_number: The port number on your device to accept REST API calls.
    Returns:
        The response of the prediction request.
    """
    # AutoML Vision Edge models will preprocess the input images.
    # The max width and height for AutoML Vision Image Classification and
    # Object Detection models are 640*480 and 1024*1024 separately. The
    # example here is for Image Classification models.
    encoded_image, height, width, img = preprocess_image(
        image_file_path=image_file_path, max_width=1024, max_height=1024)

    # The example here only shows prediction with one image. You can extend it
    # to predict with a batch of images indicated by different keys, which can
    # make sure that the responses corresponding to the given image.
    instances = {
        'instances': [
            {'image_bytes': {'b64': str(encoded_image)},
             'key': image_key}
        ]
    }

    # This example shows sending requests in the same server that you start
    # docker containers. If you would like to send requests to other servers,
    # please change localhost to IP of other servers.
    url = 'http://localhost:{}/v1/models/default:predict'.format(port_number)
    boxes = []
    try:
        response = requests.post(url, data=json.dumps(instances))
    except RequestException:
        return img, boxes
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, dict) and "predictions" in result:
            prediction = result["predictions"][0]
            assert isinstance(prediction, dict)
            detection_scores = prediction['detection_scores']
            detection_labels = prediction['detection_classes_as_text']
            detection_boxes = prediction['detection_boxes']
            for index, detection_score in enumerate(detection_scores):
                if detection_score > 0.7:
                    name = detection_labels[index]
                    detection_box = detection_boxes[index]
                    x1_percent = detection_box[1]
                    y1_percent = detection_box[0]
                    x2_percent = detection_box[3]
                    y2_percent = detection_box[2]
                    top_left = (int(x1_percent * width), int(y1_percent * height))
                    bottom_right = (int(x2_percent * width), int(y2_percent * height))
                    boxes.append((name, top_left, bottom_right))
    return img, boxes


def main():
    while True:
        img_path = questionary.text("Please input the path of the image file.").ask()
        if img_path is None:
            return
        elif os.path.isfile(img_path):
            break
    image_mat, boxes = container_predict(img_path, os.path.basename(img_path))
    for box in boxes:
        cv2.rectangle(image_mat, box[1], box[2], (255, 0, 0), thickness=1)
        cv2.putText(image_mat, box[0], box[2], cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                    thickness=2)
    cv2.imshow('result', image_mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
