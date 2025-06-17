FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv pandas numpy

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "starter.py", "./"]

RUN pipenv install --system --deploy

ENTRYPOINT ["python", "starter.py"]