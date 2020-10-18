# Detecting Number of People in Real-Time (Simulation)
In this project, I made use of [YOLOv4](https://github.com/AlexeyAB/darknet) to build a simulated real-time people detector. 
YOLOv4 is used as it offers nearly real-time and almost accurate inference when it comes to detecting objects.

In the simulation, I make use of a youtube video recorded with 30 FPS as a demonstration. As shown, the final output shows about 25 FPS, which is merely 5 FPS short, even with the dense number of people. With further tuning, or maybe a better GPU, it is possible to achieve real-time inference with YOLOv4.

Next to the video with the drawn boundary boxes, we will also have a chart to display the change in number of people in the video in real-time.

![Example of Web App](images/example.jpg?raw=true)

Possible use cases include counting the number of people in malls, especially in this covid-sensitive period, which can help relieve workloads of security guards or volunteers manning the entrances/exits.

## Requirements
- pafy
- flask
- opencv-python

Do ensure that the version of your opencv is GPU-compatible, as the model requires a GPU. CPU inference is possible, but it will drastically reduces the inference time.
Also, as the model weights are too big to fit on reddit, please follow the link below to download the model:
[yolov4.weights](https://drive.google.com/file/d/1_0z71Drjhqe-tDtJcYzKGkbA-ezZpL1L/view?usp=sharing)

## How to Run
To run the web application, simply cd into the yolov4-detect directory and type in your command line:
```python
python server.py
```

## References
- [YOLOv4 Github](https://github.com/AlexeyAB/darknet)
- [Creating Real-Time Charts with Flask](https://github.com/roniemartinez/real-time-charts-with-flask)
- [Real-time Human Detection with OpenCV](https://thedatafrog.com/en/articles/human-detection-video/)