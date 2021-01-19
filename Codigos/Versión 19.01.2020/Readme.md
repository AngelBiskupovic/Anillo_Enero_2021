#Versión 19-01-2020

Esta versión contiene 5 scripts:

-Thickener_modbus: Simula el espesador, en el se encuentra un servidor modbus que simulará el sensor fisico.

-cliente_entradas: Simula la lectura de un sensor, este script lee las entradas (2 variables).

-cliente_perturbaciones: Simula la lectura de un sensor, este script lee las perturbaciones del sistema (2 variables).

-cliente_salidas: Simula la lectura de un sensor, este script lee las entradas.

-servidor_modbus_to_opc: se encarga de juntar la información via modbus y pasarla a OPC, además se reduce la tasa de envios de 1 seg a 5 seg.
