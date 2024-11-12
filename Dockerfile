FROM python:3.11

COPY ./requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install -r /requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health