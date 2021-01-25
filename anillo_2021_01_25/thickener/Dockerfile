# set base image (host OS)
FROM anillo/base_thickener:1.0

# set the working directory in the container
WORKDIR /usr/src/app

#go to that directory
RUN cd /usr/src/app

# copy the content of the local src directory to the working directory
COPY . /usr/src/app


CMD python Thickener_modbus.py



