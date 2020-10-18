import json
import random
import time
from datetime import datetime
import cv2
import pafy
import time
import threading

from flask import Flask, Response, render_template

application = Flask(__name__)

# We count the number of people here
PEOPLE_COUNT = 0

@application.route('/')
def index():
    return render_template('index.html')

def generate_video():

    global PEOPLE_COUNT
    outputFrame = None

    CONFIDENCE_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.5
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    WEIGHTS_FILE = './model/yolov4.weights'
    CONFIG_FILE = './model/yolov4.cfg'
    CLASSES_FILE = './model/coco.names'

    class_names = []
    with open(CLASSES_FILE, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    # Initialize model for inference
    print('Initializing model...')
    net = cv2.dnn.readNet(WEIGHTS_FILE, CONFIG_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(512, 512), scale=1/255)
    print('Model initialized!')

    # URL = 'https://www.youtube.com/watch?v=8pXFktAbx5Y'

    # Video chosen is a clip of a shibuya crossing (the busiest street in Japan) walkabout
    URL = 'https://www.youtube.com/watch?v=_dWyKj7I9JM'
    video = pafy.new(URL)
    best_video = video.getbest()
    vc = cv2.VideoCapture(best_video.url)

    # We want to ensure thread-safety here
    lock = threading.Lock()

    while cv2.waitKey(1) < 1:

        (grabbed, frame) = vc.read()
        if not grabbed:
            exit()

        start = time.time()
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()

        PEOPLE_COUNT = 0

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            # only output objects with the person class
            if class_names[classid[0]] != 'person':
                continue
            # draw the boundary boxes for detected people
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            PEOPLE_COUNT += 1

        end_drawing = time.time()
        
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame = cv2.resize(frame, (720, 480))

        with lock:
            outputFrame = frame.copy()
            (flag, encodedImage) = cv2.imencode('.jpg', outputFrame)

        # turn the encoded image into a byte array
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n")

        # cv2.imshow("detections", frame)

@application.route('/video-feed')
def video_feed():
    return Response(generate_video(), mimetype = 'multipart/x-mixed-replace; boundary=frame')
    
# Draws a new point on the chart every second determing the number of people
@application.route('/chart-data')
def chart_data():
    def generate_chart():
        global PEOPLE_COUNT
        while True:
            json_data = json.dumps(
                {'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'value': PEOPLE_COUNT})
            yield f"data:{json_data}\n\n"
            time.sleep(1)

    return Response(generate_chart(), mimetype='text/event-stream')


if __name__ == '__main__':
    application.run(debug=True, threaded=True)