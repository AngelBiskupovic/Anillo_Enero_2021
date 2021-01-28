Configuración Contenedores

- Se tienen 3 carpetas de configuración (una para cada PC).
- Cada Carpeta contiene un archivo llamado file_config y un script llamado script_config_pcX.sh
-Es importante que script y el archivo json se encuentren en la misma carpeta.
-Una vez se tengan las carpetas en los pc, es importante acceder al archivo de configuración file_config.json y cambiar las direcciones IP's por la direcciones IP's a utilizar.

Para la ejecución se debe seguir el siguiente orden:

Primero ejecutar script en el pc3
-Ejecutar en consola comando chmod +x script_config_pc3.sh
-Ejecutar en consola comando ./script_config_pc3.sh

Luego ejecutar script en el pc2
-Seguir mismos pasos caso anterior chmod +x script_config_pc2.sh
-Ejecutar en consola comando ./script_config_pc2.sh

Luego ejecutar script en el pc1
-Seguir mismos pasos caso anterior chmod +x script_config.sh
-Ejecutar en consola comando ./script_config.sh
