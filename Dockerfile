FROM public.ecr.aws/lambda/python:3.11

# Install build tools
RUN yum install -y gcc gcc-c++ make && yum clean all

# Copy and install dependencies
COPY requirements-serve.txt .
RUN pip3 install --only-binary=:all: --no-cache-dir -r requirements-serve.txt

# Copy app files
COPY src/app.py ./src/app.py
COPY outputs/model.pkl ./outputs/model.pkl
COPY config.yaml .

# Override Lambda entrypoint for EC2
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]