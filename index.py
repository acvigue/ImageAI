import numpy as np
import cv2 as cv2
import requests
import hmac
from hashlib import sha1
import json
import os
from sanic import Sanic, Forbidden, Unauthorized
from sanic.response import json
import boto3

app = Sanic(name="image-annotation-server")

session = boto3.Session()

def reshape_arr(a, n): # n is number of consecutive adjacent items you want to compare for averaging
    hold = len(a)%n
    if hold != 0:
        container = a[-hold:] #numbers that do not fit on the array will be excluded for averaging
        a = a[:-hold].reshape(-1,n)
    else:
        a = a.reshape(-1,n)
        container = None
    return a, container
def get_mean(a, close): # close = how close adjacent numbers need to be, in order to be averaged together
    my_list=[]
    for i in range(len(a)):
        if a[i].max()-a[i].min() > close:
            for j in range(len(a[i])):
                my_list.append(a[i][j])
        else:
            my_list.append(a[i].mean())
    return my_list  
def final_list(a, c): # add any elemts held in the container to the final list
    if c is not None:
        c = c.tolist()
        for i in range(len(c)):
            a.append(c[i])
    return a

@app.route("/api/extractImages", methods=["POST"])
async def extractImages(request, path=""):
    if "X-Api-Signature" not in request.headers:
        raise Forbidden()
    signature = request.headers.get("X-Api-Signature", "")
    #Generate our own signature based on the request payload
    secret = os.environ.get('APP_SECRET', '').encode("utf-8")
    mac = hmac.new(secret, msg=request.body, digestmod=sha1)
    #Ensure the two signatures match
    if not str(mac.hexdigest()) == str(signature):
        raise Unauthorized()

    content = request.json
    url = content["url"]
    response = requests.get(url)

    nparr = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    laplacian = cv2.Laplacian(blur_gray,cv2.CV_8UC1, blur_gray, kernel_size)

    lsd = cv2.createLineSegmentDetector(0)
    lines_lsd = lsd.detect(laplacian)[0] #Position 0 of the returned tuple are the detected lines

    y_pos = [0]

    for line in lines_lsd:
        for x1,y1,x2,y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if abs(angle) < 0.2 and abs(abs(x2-x1)) > 300:
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 3)
                thispos = np.average([int(y1),int(y2)]);
                add = True
                for pos in y_pos:
                    if(abs(thispos-pos) < 60) :
                        add = False
                        break

                if add:
                    y_pos.append(np.average([int(y1),int(y2)]))

    y_pos.sort()

    resp = {
        "width": image.shape[1],
        "lines_found_at": y_pos
    }

    return json(resp)

@app.route("/api/annotateImage", methods=["POST"])
async def annotateImage(request, path=""):
    if "X-Api-Signature" not in request.headers:
        raise Forbidden()
    signature = request.headers.get("X-Api-Signature", "")
    #Generate our own signature based on the request payload
    secret = os.environ.get('APP_SECRET', '').encode("utf-8")
    mac = hmac.new(secret, msg=request.body, digestmod=sha1)
    #Ensure the two signatures match
    if not str(mac.hexdigest()) == str(signature):
        raise Unauthorized()
        
    try:
        content = request.json
        url = content["url"]
        response = requests.get(url)

        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        client = session.client('rekognition', region_name='us-east-1')
        response = client.detect_faces(Image={'Bytes': response.content}, Attributes=['DEFAULT'])

        resp = {
            "error": False,
            "facesCount": len(response['FaceDetails']),
            "faces": [],
            "image": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }

        for face in response['FaceDetails']:
            box = face['BoundingBox']
            x = int(box['Left'] * image.shape[1])
            y = int(box['Top'] * image.shape[0])
            w = int(box['Width'] * image.shape[1])
            h = int(box['Height'] * image.shape[0])

            face = {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }

            cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)

            resp["faces"].append(face)

        return json(resp)
    except Exception as e:
        resp = {
            "error": True,
            "backtrace": str(e)
        }

        return json(resp)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
