# set base image (host OS)
FROM python:3.8-alpine

# set the working directory in the container
WORKDIR /usr/src/app

#go to that directory
RUN cd /usr/src/app

# copy the content of the local src directory to the working directory
COPY . /usr/src/app

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

CMD python perturbations_client.py


#CMD cd pagina_web && gunicorn -w 3 -b 0.0.0.0:5000 run:app 
# command to run on container start
#CMD [ "python", "./run.py" ]
