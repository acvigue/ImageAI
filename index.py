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
from sanic.log import logger
import boto3
import urllib.parse
from functools import wraps
from playwright.async_api import async_playwright

app = Sanic(name="image-annotation-server")

session = boto3.Session()


def authorized():
    def decorator(f):
        @wraps(f)
        async def decorated_function(request, *args, **kwargs):
            if "X-Api-Signature" not in request.headers:
                logger.warn("Request without authorization!")
                return json({"status": "forbidden"}, 401)
            signature = request.headers.get("X-Api-Signature", "")
            # Generate our own signature based on the request payload
            secret = os.environ.get('APP_SECRET', '').encode("utf-8")
            mac = hmac.new(secret, msg=request.body, digestmod=sha1)
            # Ensure the two signatures match
            if not str(mac.hexdigest()) == str(signature):
                logger.error("Request with bad authorization!")
                return json({"status": "not_authorized"}, 403)

            response = await f(request, *args, **kwargs)
            return response
        return decorated_function
    return decorator


@app.route("/api/googleSearch", methods=["POST"])
@authorized()
async def googleSearch(request, path=""):
    content = request.json

    extra_params = ""
    images = False
    query = ""

    if "query" in content:
        query = content["query"]
    else:
        return json({"status": "bad_request"}, 400)

    if "extra_params" in content:
        extra_params = content["extra_params"]

    if "images" in content and content["images"] is True:
        extra_params += "&tbm=isch&tbs=isz:l"
        images = True

    googleURL = "https://google.com/search?q={}&safe=off&cr=countryUS{}".format(urllib.parse.quote_plus(query), extra_params)

    async with async_playwright() as p:
        links = []
        try:
            browser = await p.firefox.launch()
            page = await browser.new_page()
            await page.goto(googleURL)
            if images is False:
                elements = await page.locator(".MjjYud a[jsname='UWckNb']").all()
                for element in elements:
                    link = await element.get_attribute("href")
                    title = await element.locator("h3").text_content()
                    if link is not None:
                        if "/url?sa=t" in link:
                            link = link.split("&url=")[1].split("&")[0]
                            link = urllib.parse.unquote_plus(link)
                        links.append({"url": link, "title": title})
            else:
                elements = await page.locator("div[jsname='N9Xkfe']").all()
                i = 0
                for element in elements:
                    if i > 5:
                        break
                    await element.click()
                    image = await page.locator("img[jsname='kn3ccd']").get_attribute("src")
                    if image is not None:
                        links.append({"url": image, "title": "image"})
                    i = i+1
        finally:
            await browser.close()

        resp = {
            "error": False,
            "links": links
        }

        return json(resp)


@app.route("/api/extractImages", methods=["POST"])
@authorized()
async def extractImages(request, path=""):
    content = request.json

    if "url" not in content:
        return json({"status": "bad_request"}, 400)
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

        if "url" not in content:
            return json({"status": "bad_request"}, 400)
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
    
    if "url" not in content:
        return json({"status": "bad_request"}, 400)
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
    app.run(host="0.0.0.0", port=8000, access_log=True, workers=4)
