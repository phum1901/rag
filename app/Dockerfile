FROM python:3.12-slim

WORKDIR /app

RUN pip install gradio langgraph

COPY ./app.py /app

EXPOSE 7680

ENV GRADIO_SERVER_NAME="0.0.0.0"

ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
