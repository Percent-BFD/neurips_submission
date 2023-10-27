# Use an official Python runtime as a parent image
# other options in https://github.com/orgs/pytorch/packages/container/pytorch-nightly/versions?filters%5Bversion_type%5D=tagged
# Lit-GPT requires current nightly (future 2.1) for the latest attention changes
FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

# Setup requriements
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt


# Copy over single file server
COPY ./main.py /submission/main.py
COPY ./helper.py /submission/helper.py
COPY ./api.py /submission/api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
# uvicorn main:app --host 0.0.0.0 --port 8080