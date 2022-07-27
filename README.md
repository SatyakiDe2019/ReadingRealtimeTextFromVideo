# Realtime reading from a Streaming using Computer Vision

## About this app

This computer-vision app will read text from the live video or Web-CAM in real-time streaming. This application developed using Open-CV & Pytesseract. This project is for the intermediate Python developer & Data Science Newbi's. This is an extension to one of my previous installment of reading handwritten letters that was posted during the month of Jan 2022. In this post, the application will be using the model & it won't share the actual model training code.


## How to run this app

(The following instructions apply to Posix/bash. Windows users should check
[here](https://docs.python.org/3/library/venv.html).)

First, clone this repository and open a terminal inside the root folder.

Create and activate a new virtual environment (recommended) by running
the following:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the Augmented Reality-App:

```bash
python readingVideo.py
```

Make sure that you are properly connected with a functional WebCam (Preferably a separate external WebCAM) & mount that at a pre-defined distance from the subjects.

## Screenshots

![demo.GIF](demo.GIF)

## Resources

- To learn more about Open-CV, check out our [documentation](https://opencv.org/opencv-free-course/).
- To learn more about Pytesseract, check out our [documentation](https://github.com/madmaze/pytesseract).
- To view the complete demo with sound, check out our [YouTube Page](https://youtu.be/LaVVxOIUd1U).
