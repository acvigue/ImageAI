import re
import numpy as np
import cv2 as cv2
from flask import Flask, request, abort
import requests
import hmac
from hashlib import sha1
import json
import os

app = Flask(__name__)


creds = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
with open('gcreds.json', 'w') as fp:
    json.dump(creds, fp)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/extractImages", methods=["POST"])
def extractImages():
    if "X-Api-Signature" not in request.headers:
        abort(403)
    signature = request.headers.get("X-Api-Signature", "")
    # Generate our own signature based on the request payload
    secret = os.environ.get('APP_SECRET', '').encode("utf-8")
    mac = hmac.new(secret, msg=request.data, digestmod=sha1)
    # Ensure the two signatures match
    if not str(mac.hexdigest()) == str(signature):
        abort(403)

    content = request.get_json(force=True)
    url = content["url"]
    response = requests.get(url)

    file = open("image.jpg", "wb")
    file.write(response.content)
    file.close()

    image = cv2.imread("./image.jpg")
    result = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    kernel_size = 7
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 75  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    height, width, channels = image.shape
    y_vals = [0]
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y1==y2:
                y_vals.append(int(y1));

    y_vals.sort()

    resp = {
        "width": int(width),
        "lines_found_at": y_vals
    }

    return resp

@app.route("/api/annotateImage", methods=["POST"])
def annotateImage():
    if "X-Api-Signature" not in request.headers:
        abort(403)
    signature = request.headers.get("X-Api-Signature", "")
    print(signature)
    # Generate our own signature based on the request payload
    secret = os.environ.get('APP_SECRET', '').encode("utf-8")
    mac = hmac.new(secret, msg=request.data, digestmod=sha1)
    # Ensure the two signatures match
    if not str(mac.hexdigest()) == str(signature):
        abort(403)
        
    try:
        content = request.get_json(force=True)
        url = content["url"]
        response = requests.get(url)

        file = open("image.jpg", "wb")
        file.write(response.content)
        file.close()

        image = cv2.imread("./image.jpg")
        height, width, channels = image.shape
        
        from google.cloud import vision
        client = vision.ImageAnnotatorClient.from_service_account_json("gcreds.json")

        image = vision.Image()
        image.source.image_uri = url

        response = client.face_detection(image=image)
        
        faces = response.face_annotations
        if "error" in response:
            resp = {
                "error": True,
                "is_gcp_vision_error": True,
                "code": response.error.code,
                "message": response.error.message
            }
            return resp

        resp = {
            "error": False,
            "facesCount": len(faces),
            "faces": [],
            "image": {
                "width": width,
                "height": height
            }
        }

        for face in faces:
            x = face.bounding_poly.vertices[0].x;
            y = face.bounding_poly.vertices[0].y;
            w = face.bounding_poly.vertices[1].x - x;
            h = face.bounding_poly.vertices[2].y - y;

            face = {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }

            resp["faces"].append(face);

        return resp
    except Exception as e:
        resp = {
            "error": True,
            "backtrace": str(e)
        }

        return resp
