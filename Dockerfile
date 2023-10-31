FROM mcr.microsoft.com/playwright/python:latest-arm64

LABEL mantainer="Aiden Vigue <aiden@vigue.me>"

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "python3", "-m" , "sanic", "index:app", "-H", "0.0.0.0"]