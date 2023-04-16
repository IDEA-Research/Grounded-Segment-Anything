build-image:
	docker build -t=gsa:v0 .

run: build-image
	nvidia-docker run --gpus all -it --rm --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=$DISPLAY \
	-v ${PWD}:/home/appuser \
	--name=gsa \
	--ipc=host -it gsa:v0
