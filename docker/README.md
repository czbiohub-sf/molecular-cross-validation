This builds a Docker image that can be used to run all the code in this repo. The required packages are listed below&emdash;their dependencies are omitted for brevity.

 * miniconda with Python 3.7
 * jupyterlab
 * PyTorch
 * numpy
 * scanpy (including dependencies listed [here](https://scanpy.readthedocs.io/en/latest/installation.html))
 * magic-impute
 * altair
 * pandas
 * scikit-learn
 * umap-learn
 * and [simscity](https://www.github.com/czbiohub/simscity) for simulating single-cell data

Dependencies for building the docker images:
 - **docker** (>= version 17.05)
 - **nvidia-docker** Refer to the [readme](https://github.com/NVIDIA/nvidia-docker) for installation.


To build the image with GPU support:

```sh
nvidia-docker build -t mcv-gpu-3.7 \
	--build-arg uid=$(id -u) \
	--build-arg gid=$(id -g) \
	--build-arg cuda=1 \
	--build-arg python_version=3.7 \
	-f Dockerfile .
```

To build a CPU version:

```sh
docker build -t mcv-cpu-3.7 \
	--build-arg uid=$(id -u) \
	--build-arg gid=$(id -g) \
	--build-arg cuda=0 \
	--build-arg python_version=3.7 \
	-f Dockerfile .
```
