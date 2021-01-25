Archivos source 1.0

a) para crear el contenedor, ejecutar el script build.sh en cada carpeta. 

b) El orden para crear los "n" contenedores son:

	1) base_thickener ( no es necesario correr esta imagen una vez creada )
	2) thickener
	3..n) el resto

para correr las imagenes desde 2) a 3..n)  utilizar el siguiente comando docker:

docker run -it --network=host --name <cualquier_nombre> anillo/<nombre_imagen>:1.0 


El orden para correr las imágenes son:

1) docker run -it --network=host --name opc_server anillo/opc_server_pc3:1.0

2) docker run -it --network=host --name modbus_to_opc anillo/server_modbus_to_opc_pc2:1.0

3) docker run -it --network=host --name thickener anillo/thickener_modbus:1.0

4) docker run -it --network=host --name inputs_client anillo/sensors/inputs_client:1.0

5) docker run -it --network=host --name outputs_client anillo/sensors/outputs_client:1.0

6) docker run -it --network=host --name perturbations_client anillo/sensors/perturbations_client:1.0

7) docker run -it --network=host --name opc_controller anillo/opc_controller_pc3:1.0

