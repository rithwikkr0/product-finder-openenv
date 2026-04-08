FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install openai pydantic

CMD ["python","inference.py"]
