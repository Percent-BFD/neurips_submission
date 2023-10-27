# Make your GPUs visible to Docker 
Follow this guide to install [nvidia-ctk](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
```sh
nvidia-ctk runtime configure
systemctl restart docker
```

# Build and run 
```sh
docker build -t toy_submission .
docker run --gpus "device=0" -p 8080:80 toy_submission
```
# Send requests
```sh
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "The capital of france is "}' http://localhost:8080/process
```
