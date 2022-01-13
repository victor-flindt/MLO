FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY models/ models/
COPY references/ references/

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]