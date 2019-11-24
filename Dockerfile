FROM python:3.6-slim
COPY ./app.py /deploy/
COPY ./models.py /deploy/
COPY ./utils.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./model.h5 /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]
