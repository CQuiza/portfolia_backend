FROM python:3.12-slim

WORKDIR /api

COPY requirements.txt .
RUN pip install uv
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .


EXPOSE 8000