FROM python:3.10.4

RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get install -y ant && \
    apt-get clean;

RUN useradd -ms /bin/bash python-api
USER python-api

WORKDIR /repo
COPY . /repo

RUN python -m pip install -r requirements.txt

EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["AssignmentCS.py"]
