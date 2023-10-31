import numpy as np
import cv2 as cv2
import requests
import hmac
from hashlib import sha1
import json
import os
from scipy.signal import find_peaks
from sanic import Sanic
from sanic.response import json
import boto3
from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib.parse
from functools import wraps

app = Sanic(name="image-annotation-server")

session = boto3.Session()


def authorized():
    def decorator(f):
        @wraps(f)
        async def decorated_function(request, *args, **kwargs):
            if "X-Api-Signature" not in request.headers:
                return json({"status": "forbidden"}, 401)
            signature = request.headers.get("X-Api-Signature", "")
            # Generate our own signature based on the request payload
            secret = os.environ.get('APP_SECRET', '').encode("utf-8")
            mac = hmac.new(secret, msg=request.body, digestmod=sha1)
            # Ensure the two signatures match
            if not str(mac.hexdigest()) == str(signature):
                return json({"status": "not_authorized"}, 403)

            response = await f(request, *args, **kwargs)
            return response
        return decorated_function
    return decorator

@app.route("/api/googleSearch", methods=["POST"])
@authorized()
async def googleSearch(request, path=""):
    content = request.json
    query = content["query"]
    isch = content["extra_params"]

    option = webdriver.ChromeOptions()
    option.add_argument("window-size=1280,800")
    option.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36")

    driver = webdriver.Remote(
        command_executor=os.environ.get('GRID_HOST', ''), options=option)

    links = []
    try:
        driver.get(
            "https://google.com/search?q={}&safe=images&cr=countryUS{}".format(urllib.parse.quote_plus(query), isch))
        elements = driver.find_elements(
            By.CSS_SELECTOR, ".MjjYud a[jsname='UWckNb']")
        for element in elements:
            link = element.get_attribute("href")
            if link is not None:
                if "/url?sa=t" in link:
                    link = link.split("&url=")[1].split("&")[0]
                    link = urllib.parse.unquote_plus(link)
                links.append(link)
    finally:
        driver.quit()

    resp = {
        "error": False,
        "links": links
    }

    return json(resp)


@app.route("/api/extractImages", methods=["POST"])
@authorized()
async def extractImages(request, path=""):
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
@authorized()
async def annotateImage(request, path=""):
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
@authorized()
async def imageOrientation(request, path=""):
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
