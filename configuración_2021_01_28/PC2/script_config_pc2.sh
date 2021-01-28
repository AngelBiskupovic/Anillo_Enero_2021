#!/bin/bash

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

image_name_servidor_modbus=`jq '.Containers[1].Imagen_name' file_config.json`
IP_container_servidor_modbus=`jq '.Containers[1].Configuracion.IP_adress_container' file_config.json`
Port_container_servidor_modbus=`jq '.Containers[1].Configuracion.Port_needed[0]' file_config.json`
let size_image_servidor_modbus=${#image_name_servidor_modbus}-2
image_name_servidor_modbus=${image_name_servidor_modbus:1:$size_image_servidor_modbus}
let size_image_port_servidor_modbus=${#Port_container_servidor_modbus}-2
Port_container_servidor_modbus=${Port_container_servidor_modbus:1:$size_image_port_servidor_modbus}

docker pull $image_name_servidor_modbus
docker run -it --ip=$IP_container_servidor_modbus -p $Port_container_servidor_modbus:$Port_container_servidor_modbus -v $(pwd):/data --name modbus_to_opc $image_name_servidor_modbus
