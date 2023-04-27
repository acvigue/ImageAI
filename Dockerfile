ARG PYTHON_VERSION="3.11.3"
FROM python:${PYTHON_VERSION}-slim-buster

LABEL mantainer="Aiden Vigue <aiden@vigue.me>"

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "python3", "-m" , "sanic", "index:app"]
