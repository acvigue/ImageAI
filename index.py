import numpy as np
import cv2 as cv2
import requests
import hmac
from hashlib import sha1
import json
import os
from scipy.signal import find_peaks
from sanic import Sanic, Forbidden, Unauthorized
from sanic.response import json
import boto3

app = Sanic(name="image-annotation-server")

session = boto3.Session()


@app.route("/api/extractImages", methods=["POST"])
async def extractImages(request, path=""):
    if "X-Api-Signature" not in request.headers:
        raise Forbidden()
    signature = request.headers.get("X-Api-Signature", "")
    # Generate our own signature based on the request payload
    secret = os.environ.get('APP_SECRET', '').encode("utf-8")
    mac = hmac.new(secret, msg=request.body, digestmod=sha1)
    # Ensure the two signatures match
    if not str(mac.hexdigest()) == str(signature):
        raise Unauthorized()

    content = request.json
    url = content["url"]
    response = requests.get(url)

    nparr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    blurred = cv2.bilateralFilter(img, 10, 40, 50)
    edges = cv2.Canny(blurred, 0, 255)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))

    h_morphed = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=2)
    h_morphed = cv2.dilate(h_morphed, None)

    h_acc = cv2.reduce(h_morphed, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    h_peaks, h_props = find_peaks(
        h_acc[:, 0], 0.50*max(h_acc[:, 0]), None, 100)

    last_peak = 0
    peaks = [0]

    for peak_index in h_peaks:
        if peak_index - last_peak > (img.shape[1] / 2):
            peaks.append(int(peak_index))
            last_peak = peak_index

    resp = {
        "error": False,
        "image": {
            "width": img.shape[1],
            "height": img.shape[0]
        },
        "splits": peaks
    }

    return json(resp)


@app.route("/api/extractFaces", methods=["POST"])
async def annotateImage(request, path=""):
    if "X-Api-Signature" not in request.headers:
        raise Forbidden()
    signature = request.headers.get("X-Api-Signature", "")
    # Generate our own signature based on the request payload
    secret = os.environ.get('APP_SECRET', '').encode("utf-8")
    mac = hmac.new(secret, msg=request.body, digestmod=sha1)
    # Ensure the two signatures match
    if not str(mac.hexdigest()) == str(signature):
        raise Unauthorized()

    try:
        content = request.json
        url = content["url"]
        response = requests.get(url)

        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        client = session.client('rekognition', region_name='us-east-1')
        response = client.detect_faces(
            Image={'Bytes': response.content}, Attributes=['DEFAULT'])

        resp = {
            "error": False,
            "faces": [],
            "image": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }

        for face in response['FaceDetails']:
            box = face['BoundingBox']
            x = round(int(box['Left'] * image.shape[1]))
            y = round(int(box['Top'] * image.shape[0]))
            w = round(int(box['Width'] * image.shape[1]))
            h = round(int(box['Height'] * image.shape[0]))

            face = {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }

            resp["faces"].append(face)

        return json(resp)
    except Exception as e:
        resp = {
            "error": True,
            "backtrace": str(e)
        }

        return json(resp)


@app.route("/api/imageOrientation", methods=["POST"])
async def imageOrientation(request, path=""):
    if "X-Api-Signature" not in request.headers:
        raise Forbidden()
    signature = request.headers.get("X-Api-Signature", "")
    # Generate our own signature based on the request payload
    secret = os.environ.get('APP_SECRET', '').encode("utf-8")
    mac = hmac.new(secret, msg=request.body, digestmod=sha1)
    # Ensure the two signatures match
    if not str(mac.hexdigest()) == str(signature):
        raise Unauthorized()

    content = request.json
    url = content["url"]
    response = requests.get(url)

    nparr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    resp = {
        "error": False,
        "image": {
            "width": img.shape[1],
            "height": img.shape[0]
        }
    }

    return json(resp)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
