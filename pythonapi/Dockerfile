FROM python:3.7-stretch
WORKDIR /deploy/
COPY ./requirements.txt /deploy/
COPY ./app.py /deploy/
COPY ./score_objects.pkl /deploy/
COPY ./helper.py /deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]
