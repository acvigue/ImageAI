FROM hdgigante/python-opencv:4.7.0-alpine

WORKDIR /app

RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "python3", "-m" , "sanic", "index:app"]
