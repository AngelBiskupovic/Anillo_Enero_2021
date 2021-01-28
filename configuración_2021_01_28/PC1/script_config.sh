#!/bin/bash

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

########################Lectura de variables desde file_config.json############################################

#Se obtienen nombres de imagenes a descargar
image_name_opc=`jq '.Containers[0].Imagen_name' file_config.json`
image_name_servidor_modbus=`jq '.Containers[1].Imagen_name' file_config.json`
image_name_inputs=`jq '.Containers[2].Imagen_name' file_config.json`
image_name_perturbations=`jq '.Containers[3].Imagen_name' file_config.json`
image_name_outputs=`jq '.Containers[4].Imagen_name' file_config.json`
image_name_controller=`jq '.Containers[5].Imagen_name' file_config.json`

#Se obtienen direcciones IP y puerto de contenedores
IP_container_opc=`jq '.Containers[0].Configuracion.IP_adress_container' file_config.json`
Port_container_opc=`jq '.Containers[0].Configuracion.Port_needed' file_config.json`

IP_container_servidor_modbus=`jq '.Containers[1].Configuracion.IP_adress_container' file_config.json`
Port_container_servidor_modbus=`jq '.Containers[1].Configuracion.Port_needed' file_config.json`

IP_container_inputs=`jq '.Containers[2].Configuracion.IP_adress_container' file_config.json`
Port_container_inputs=`jq '.Containers[2].Configuracion.Port_needed[0]' file_config.json`

IP_container_perturbations=`jq '.Containers[3].Configuracion.IP_adress_container' file_config.json`
Port_container_perturbations=`jq '.Containers[3].Configuracion.Port_needed[0]' file_config.json`

IP_container_outputs=`jq '.Containers[4].Configuracion.IP_adress_container' file_config.json`
Port_container_outputs=`jq '.Containers[4].Configuracion.Port_needed[0]' file_config.json`

#Se adaptan variables para no tener error de formato en docker
let size_image_opc=${#image_name_opc}-2
image_name_opc=${image_name_opc:1:$size_image_opc}
let size_image_servidor_modbus=${#image_name_servidor_modbus}-2
image_name_servidor_modbus=${image_name_servidor_modbus:1:$size_image_servidor_modbus}
let size_image_inputs=${#image_name_inputs}-2
image_name_inputs=${image_name_inputs:1:$size_image_inputs}
let size_image_perturbations=${#image_name_perturbations}-2
image_name_perturbations=${image_name_perturbations:1:$size_image_perturbations}
let size_image_outputs=${#image_name_outputs}-2
image_name_outputs=${image_name_outputs:1:$size_image_outputs}
let size_image_controller=${#image_name_controller}-2
image_name_controller=${image_name_controller:1:$size_image_controller}
let size_image_port_opc=${#Port_container_opc}-2
Port_container_opc=${Port_container_opc:1:$size_image_port_opc}
let size_image_port_servidor_modbus=${#Port_container_servidor_modbus}-2
Port_container_servidor_modbus=${Port_container_servidor_modbus:1:$size_image_port_servidor_modbus}
let size_image_port_inputs=${#Port_container_inputs}-2
Port_container_inputs=${Port_container_inputs:1:$size_image_port_inputs}
let size_image_port_perturbations=${#Port_container_perturbations}-2
Port_container_perturbations=${Port_container_perturbations:1:$size_image_port_perturbations}
let size_image_port_outputs=${#Port_container_outputs}-2
Port_container_outputs=${Port_container_outputs:1:$size_image_port_outputs}

###########################Descarga de Imágenes desde docker hub################################################

#docker pull $image_name_opc
#docker pull $image_name_servidor_modbus
docker pull $image_name_inputs
docker pull $image_name_perturbations
docker pull $image_name_outputs
#docker pull $image_name_controller

#######################################Ejecución de contenedores################################################

#docker run -d --ip=$IP_container_servidor_modbus -p $Port_container_servidor_modbus:$Port_container_servidor_modbus -v $(pwd):/data --name modbus_to_opc $image_name_servidor_modbus & sleep 3
docker run -d --network=host -v $(pwd):/data --name thickener anillo/thickener_modbus:1.0 & sleep 3
docker run -d --ip=$IP_container_inputs  -v $(pwd):/data --name inputs_client $image_name_inputs & sleep 3
docker run -d --ip=$IP_container_perturbations -v $(pwd):/data --name perturbations_client $image_name_perturbations & sleep 3
docker run -d --ip=$IP_container_outputs  -v $(pwd):/data --name outputs_client $image_name_outputs & sleep 3
#docker run -d --network=host -v $(pwd):/data --name opc_controller anillo/opc_controller_pc3:1.0

#docker run -it --ip=$IP_container_opc -p $Port_container_opc:$Port_container_opc -v $(pwd):/data --name opc_server $image_name_opc
#docker run -d --network=host --name modbus_to_opc anillo/server_modbus_to_opc_pc2:1.0 & sleep 3
#docker run -d --network=host --name thickener anillo/thickener_modbus:1.0 & sleep 3
#docker run -d --network=host --name inputs_client anillo/sensors/inputs_client:1.0 & sleep 3
#docker run -d --network=host --name outputs_client anillo/sensors/outputs_client:1.0 & sleep 3
#docker run -d --network=host --name perturbations_client anillo/sensors/perturbations_client:1.0 & sleep 3
#docker run -d --network=host --name opc_controller anillo/opc_controller_pc3:1.0  --ip=$variable