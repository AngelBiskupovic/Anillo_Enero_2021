#Versión 2021-01_09 MT

Esta versión contiene 3 scripts:

-OPC_Client: Clase llamada por el script "servidor_modbus_server_to_opc_pc2.py" 
que contiene el Cliente OPCUA, modificado de la versión de Saúl (sin SubHandler).
No es necesario correr este script.


1) opcua_server_pc3: Servidor OPCUA que inicializa los nodos y objetos que serán escritos/leidos por el cliente

2) servidor_modbus_to_opc_pc2: se encarga de juntar la información via modbus y pasarla a OPC, además se reduce la tasa de envios de 1 seg a 5 seg.



Contiene 1 librería extra:

pip3 install opcua

Nota: Correr primero el 1) y despúes el 2)
Nota2: Se puede reemplazar localhost por la dirección IP que necesites. tampoco requiere de conexión ethernet necesariamente. 

 
