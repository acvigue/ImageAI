import re
import numpy as np
import cv2 as cv2
import requests
import hmac
from hashlib import sha1
from json import dumps, loads
import os
from scipy.signal import find_peaks
from sanic import Sanic
from sanic.response import json, redirect
from sanic.log import logger
import boto3
import urllib.parse
from functools import wraps
from bs4 import BeautifulSoup

app = Sanic(name="image-annotation-server")
b3 = boto3.Session()
session = requests.session()

#HMAC decorator
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

#dirty knees!
@app.route("/", methods=["GET"])
async def index(request, path=""):
    return redirect("https://www.youtube.com/watch?v=FfnQemkjPjM")

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

    logger.info('searching for "{}" (images {})'.format(query, images))

    googleURL = "https://google.com/search?q={}&safe=off&cr=countryUS{}".format(
        urllib.parse.quote_plus(query), extra_params)

    # we are not a robot
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    req = None
    req = session.get(googleURL, headers=headers)

    # Check if we had encountered the consent popup, if so, deal w/ it
    if "<title>Before you continue</title>" in req.text:
        logger.warn('Encountered consent page')
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        reqData = """gl=DE&m=0&app=0&pc=irp&continue={}&x=6&bl=boq_identityfrontenduiserver_20231024.06_p0&hl=en-US&src=1&cm=2&set_sc=true&set_eom=false&set_aps=true""".format(
            urllib.parse.quote_plus(googleURL))
        req = session.post("https://consent.google.com/save",
                           reqData, headers=headers)
        logger.info('Consent cookies generated and saved to session')
    resp = req.text

    # soup is now good
    bs = BeautifulSoup(resp, features="lxml")

    links = []
    if images is False:
        results = bs.select(".MjjYud a[jsname='UWckNb']")
        for result in results:
            link = result["href"]
            title = result.find("h3").text
            if link is not None:
                if link.find("/url?") == 0:
                    link = link.split("&url=")[1].split("&")[0]
                    link = urllib.parse.unquote_plus(link)
                links.append({"url": link, "title": title})
    else:
        all_script_tags = bs.select("script")
        matched_images_data = "".join(re.findall(
            r"AF_initDataCallback\(([^<]+)\);", str(all_script_tags)))
        matched_images_data_fix = dumps(matched_images_data)
        matched_images_data_json = loads(matched_images_data_fix)
        matched_google_image_data = re.findall(
            r'\"b-GRID_STATE0\"(.*)sideChannel:\s?{}}', matched_images_data_json)
        removed_matched_google_images_thumbnails = re.sub(
            r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]', "", str(matched_google_image_data))
        matched_google_full_resolution_images = re.findall(
            r"(?:'|,),\[\"(https:|http.*?)\",\d+,\d+\]", removed_matched_google_images_thumbnails)
        full_res_images = [
            bytes(bytes(img, "ascii").decode("unicode-escape"), "ascii").decode("unicode-escape") for img in matched_google_full_resolution_images
        ]

        for img in full_res_images:
            links.append({"url": img, "title": ""})

    logger.info("Got {} results".format(len(links)))

    resp = {
        "error": False,
        "links": links[0:10]
    }

    return json(resp)


@app.route("/api/extractImages", methods=["POST"])
@authorized()
async def extractImages(request, path=""):
    content = request.json

    if "url" not in content:
        return json({"status": "bad_request"}, 400)
    url = content["url"]

    logger.info('Extracting images from {}'.format(url))

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

    logger.info("Found {} splits".format(len(peaks) - 1))

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

        logger.info('Extracting faces from {}'.format(url))

        response = requests.get(url)

        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        client = b3.client('rekognition', region_name='us-east-1')
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

        logger.info("Located {} faces".format(len(resp["faces"])))

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

    logger.info('Getting dimensions of {}'.format(url))

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

    logger.info("Image is {}x{}".format(img.shape[1], img.shape[0]))

    return json(resp)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, access_log=True)
