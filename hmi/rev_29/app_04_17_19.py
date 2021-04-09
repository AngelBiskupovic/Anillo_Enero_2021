# -*- coding: utf-8 -*-
from flask import Flask, jsonify, render_template, request, flash, redirect, session, abort, url_for
from cassandra.cluster import Cluster
from cassandra import timestamps
import time, datetime
from bokeh.embed import server_document, components
from bokeh.resources import INLINE
from bokeh.plotting import curdoc, figure
from bokeh.layouts import row, column, gridplot
from bokeh.models.sources import ColumnDataSource,AjaxDataSource
from bokeh.models import HoverTool, DatetimeTickFormatter,Legend
from bokeh.core.json_encoder import serialize_json
from bokeh.util.serialization import transform_column_source_data, transform_series


from datetime import  timezone
import pandas as pd
from numpy import fft
import numpy as np
import random
import math
import uuid
import os
import pytz
from werkzeug.security import check_password_hash

resources = INLINE
js_resources = resources.render_js()
css_resources = resources.render_css()

app = Flask(__name__)

# To prevent caching files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or \
                           'a3ac388c-d1cf-31e5-6e39-f3f433c10b74'

def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)

# variables gobales
x, y = 0, 0

global connection

ACCESS = {
    'login': 0,
    'op': 1,
    'dev': 2,
    'admin': 3
}


class User():
# Se define la clase "User" para manejar y revisar los accesos de los usuarios

    def __init__(self, name, password, access=ACCESS['login'], cookie=0):
	# Se inician los parámetros que tendrá cada usuario, por defecto se inician un usuario sin permisos	
		
        self.name = name
        self.password = password
        self.access = access
        self.cookie = cookie
	
    def is_login_valid(username, password):
	# Esta función revisa si los datos ingresados en la página de login son válidos para iniciar sesión
	
		# Se busca en la tabla "usuarios.data" los datos del usuario (si existiese)
        rows = connection_usr.execute("select * from usuarios.data where name = '" + username + "'")
        df = rows._current_rows
        
        # Si se encuentra un usuario con el nombre se extrae el hash de la contraseña y se compara con la ingresada
        if not df.empty:
			
			# Nombre obtenido de la tabla
            name = df.iloc[:, 0].item()
            
            # Contraseña obtenida de la tabla
            password_csq = df.iloc[:, 2].item()
            
            # Se comprueba si los datos ingresados coinciden con los de la tabla
            return (name == username) and check_password_hash(password_csq, password)  
		
		# Si no hay datos del usuario automáticamente el acceso se invalida
        else:
            return False

    def find_by_username(username):
	# 
	
		# Se busca en la tabla "usuarios.data" los datos del usuario (si existiese)
        rows = connection_usr.execute("select * from usuarios.data where name = '" + username + "'")
        df = rows._current_rows
        
        # Nombre del usuario
        name = df.iloc[:, 0].item()
        
        # Nivel de acceso del usuario
        access = df.iloc[:, 1].item()
        
        # Hash de la contraseña del usuario
        password = df.iloc[:, 2].item()
        
        # Nuevo identificador del usuario (generado de forma aleatoria)
        session_id = uuid.uuid4()

		# Se actualiza el identificador del usuario con el nuevo 
        connection_usr.execute(
            "UPDATE usuarios.data SET session_id=" + str(session_id) + " WHERE name = '" + name + "' and access=" + str(
                access))
        
        return User(str(name), str(password), access, session_id)

    def access_by_sessionid(userid):
		
		# Se extrae de la tabla "usuarios.data" los datos de todos los usuarios
        rows = connection_usr.execute("select * from usuarios.data")
        df = rows._current_rows
        
        # Si hay datos se comprueba el nivel de acceso asociado al identificador del usuario (userid)
        if not df.empty:
			
			# Identificadores de los usuarios
            session_id = df.iloc[:, 3]
	
			# Se busca el nivel de acceso del usuario
            for i in range(len(session_id)):
                if session_id[i] == userid:
                    return df.iloc[:, 1][i]
		# En caso de no encontrar el nivel de acceso del usuario se le asigna el menor posible
        return 0

###########################
# Se habilita el acceso al keyspace "usuarios"

cluster_usr = Cluster()
keyspace_usr = 'usuarios'
connection_usr = cluster_usr.connect(keyspace_usr)
connection_usr.row_factory = pandas_factory
connection_usr.default_fetch_size = None

###########################
# Se habilita el acceso al keyspace "fondef"

cluster = Cluster()
keyspace = 'fondef'
connection = cluster.connect(keyspace)
connection.row_factory = pandas_factory
connection.default_fetch_size = None

# tags de todos los sensores necesarios para la pestaña "Espesador"
todosTags = ["bf_7110_dt_1011", "bg_7110_dt_1011_solido", "bh_7110_dt_1030", "bi_7110_dt_1030_solido",
             "bj_7110_ft_1012",
             "bk_7110_ft_1030", "bl_7110_lt_1009", "bm_7110_lt_1009_s2", "bn_7110_lt_1009_s3", "bo_7110_lt_1009_s4",
             "bp_7110_ot_1003", "bq_7110_pt_1010", "br_7120_ft_1002", "bs_7120_ft_1001", "bt_7120_lit_001",
             "bu_4310_pt_0022", "bv_4310_pt_0021", "bw_4310_pt_1030", "bx_delta_de_presion", "by_7120_ft_0014",
             "bz_7120_pk_timer_ciclo_on", "ca_7120_pk_timer_ciclo_off", "cb_7120_fic_1002_sp_ext",
             "cc_4310_pu_0003_sp_ext",
             "cd_4310_pu_0004_sp_ext", "ce_4310_pu_0003_vstatus", "cf_4310_pu_0004_vstatus","ci_4310_pt_0023"]

todosTags_control = ['bi_7110_dt_1030_solido_inf', 'bi_7110_dt_1030_solido_lamb','bi_7110_dt_1030_solido_q',
					 'bi_7110_dt_1030_solido_sp','bi_7110_dt_1030_solido_sup', 'bi_7110_dt_1030_solido_t',
					 'bk_7110_ft_1030_dinf', 'bk_7110_ft_1030_dsup', 'bk_7110_ft_1030_inf',
					 'bk_7110_ft_1030_r', 'bk_7110_ft_1030_sup', 'bo_7110_lt_1009_s4_inf',
					 'bo_7110_lt_1009_s4_lamb', 'bo_7110_lt_1009_s4_q', 'bo_7110_lt_1009_s4_sp',
					 'bo_7110_lt_1009_s4_sup','bo_7110_lt_1009_s4_t', 'bp_7110_ot_1003_inf', 
					 'bp_7110_ot_1003_lamb','bp_7110_ot_1003_q', 'bp_7110_ot_1003_sp', 
					 'bp_7110_ot_1003_sup','bp_7110_ot_1003_t', 'bq_7110_pt_1010_inf', 
					 'bq_7110_pt_1010_lamb','bq_7110_pt_1010_q', 'bq_7110_pt_1010_sp', 
					 'bq_7110_pt_1010_sup','bq_7110_pt_1010_t', 'br_7120_ft_1002_dinf', 
					 'br_7120_ft_1002_dsup','br_7120_ft_1002_inf', 'br_7120_ft_1002_r', 'br_7120_ft_1002_sup']

@app.route('/_get_value', methods=['POST'])
def get_value():

#	Esta función obtiene los valores de todos los sensores que se muestran en la visual inicial de la pestaña "Espesador", los valores se obtienen
#   con peticiones a la tabla "fondef.dcs_sensors".

    try:

        if request.method == 'POST':
			
			# Variable que guardará los datos de los sensores
            df_tot = [0] * len(todosTags)

			# Peticiones del valor más reciente de cada sensor a la tabla de "fondef.dcs_sensors" de cassandra
            for r in range(0, len(todosTags)):
                rows = connection.execute(
                    "select * from fondef.dcs_sensors where sensor = '{}' LIMIT 1".format(todosTags[r]))
                df = rows._current_rows
                df = df.iloc[::-1]
                df = df.reset_index(drop=True)
                df_tot[r] = df

            df_tot = pd.concat(df_tot, ignore_index=True)
            df_tot = pd.pivot_table(df_tot, values='value', columns='sensor', index='datetime')
            df_tot = df_tot.reset_index(drop=False)

            df_tot['datetime'] = df_tot['datetime'].dt.tz_localize('utc').dt.tz_convert('America/Caracas')
            df_tot['datetime'] = df_tot['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

			# Variable que guarda la fecha y hora de los datos
            datetime = df_tot["datetime"].iloc[-1]
            df_tot = df_tot.set_index('datetime', drop=True)
            df_tot = df_tot.reset_index(drop=True)
			
			# Asignación de variables a cada valor de cada sensor 
            bf_7110_DT_1011 = df_tot['bf_7110_dt_1011'].iloc[-1]
            bg_7110_DT_1011_SOLIDO = df_tot['bg_7110_dt_1011_solido'].iloc[-1]
            bh_7110_DT_1030 = df_tot['bh_7110_dt_1030'].iloc[-1]
            bi_7110_DT_1030_SOLIDO = df_tot['bi_7110_dt_1030_solido'].iloc[-1]
            bj_7110_FT_1012 = df_tot['bj_7110_ft_1012'].iloc[-1]
            bk_7110_FT_1030 = df_tot['bk_7110_ft_1030'].iloc[-1]
            bl_7110_LT_1009 = df_tot['bl_7110_lt_1009'].iloc[-1]
            bm_7110_LT_1009_S2 = df_tot['bm_7110_lt_1009_s2'].iloc[-1]
            bn_7110_LT_1009_S3 = df_tot['bn_7110_lt_1009_s3'].iloc[-1]
            bo_7110_LT_1009_S4 = df_tot['bo_7110_lt_1009_s4'].iloc[-1]
            bp_7110_OT_1003 = df_tot['bp_7110_ot_1003'].iloc[-1]
            bq_7110_PT_1010 = df_tot['bq_7110_pt_1010'].iloc[-1]
            br_7120_FT_1002 = df_tot['br_7120_ft_1002'].iloc[-1]
            bs_7120_FT_1001 = df_tot['bs_7120_ft_1001'].iloc[-1]
            bt_7120_LIT_001 = df_tot['bt_7120_lit_001'].iloc[-1]      
            ci_4310_PT_0023 = df_tot['ci_4310_pt_0023'].iloc[-1]
            
			
            bu_ph_pozo = random.uniform(1, 15)
            bv_conduct_pozo = random.uniform(0, 1500)
            bw_temp_pozo = random.uniform(5, 30)
            bx_ph_esp = random.uniform(1, 15)
            by_conduct_esp = random.uniform(0, 1500)

		# Valor que ajusta el número de decimales mostrados en la pantalla
        decimal = 2
		
		# Se retornan los valores redondeados en formato json 
        return jsonify(datetime=datetime, bf_7110_DT_1011=round(bf_7110_DT_1011, decimal),
                       bg_7110_DT_1011_SOLIDO=round(bg_7110_DT_1011_SOLIDO, decimal),
                       bh_7110_DT_1030=round(bh_7110_DT_1030, decimal),
                       bi_7110_DT_1030_SOLIDO=round(bi_7110_DT_1030_SOLIDO, decimal),
                       bj_7110_FT_1012=round(bj_7110_FT_1012, decimal),
                       bk_7110_FT_1030=round(bk_7110_FT_1030, decimal), bl_7110_LT_1009=round(bl_7110_LT_1009, decimal),
                       bm_7110_LT_1009_S2=round(bm_7110_LT_1009_S2, decimal),
                       bn_7110_LT_1009_S3=round(bn_7110_LT_1009_S3, decimal),
                       bo_7110_LT_1009_S4=round(bo_7110_LT_1009_S4, decimal),
                       bp_7110_OT_1003=round(bp_7110_OT_1003, decimal),
                       bq_7110_PT_1010=round(bq_7110_PT_1010, decimal), br_7120_FT_1002=round(br_7120_FT_1002, decimal),
                       bs_7120_FT_1001=round(bs_7120_FT_1001, decimal), bt_7120_LIT_001=round(bt_7120_LIT_001, decimal),
                       bu_ph_pozo=round(bu_ph_pozo, decimal), bv_conduct_pozo=round(bv_conduct_pozo, decimal),
                       bw_temp_pozo=round(bw_temp_pozo, decimal), bx_ph_esp=round(bx_ph_esp, decimal),
                       ci_4310_PT_0023=round(ci_4310_PT_0023, decimal),by_conduct_esp=round(by_conduct_esp, decimal))
    
    except Exception as e:
     return str(e)

@app.route('/_get_control_value', methods=['POST'])
def get_control_value():

# Esta función obtiene los parámetros que ajustan el comportamiento del controlador, los valores se obtienen de la tabla "fondef.set_mpc_controller".

    try:
		# Nombre de los tags necesarios para ajustar al controlador se obtienen de la variable "todosTags_control"

        if request.method == 'POST':
			
			# Variable que guardará los datos de los parámetros
            df_tot = [0] * len(todosTags_control)

			# Peticiones del valor más reciente de cada sensor a la tabla de "fondef.mpc_weights" de cassandra
            for r in range(0, len(todosTags_control)):
                rows = connection.execute(
                    "select * from fondef.mpc_weights where sensor = '{}' LIMIT 1".format(todosTags_control[r]))
                df = rows._current_rows
                df = df.iloc[::-1]
                df = df.reset_index(drop=True)
                df_tot[r] = df
                

            df_tot = pd.concat(df_tot, ignore_index=True)
            
            df_tot = pd.pivot_table(df_tot, values='value', columns='sensor')
            df_tot = df_tot.reset_index(drop=False)
            
            # Variable que guarda la fecha y hora de los datos
            date_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            
            # Asignación de variables a cada valor de cada parámetro 
            bi_7110_dt_1030_solido_inf  = df_tot['bi_7110_dt_1030_solido_inf'].iloc[-1]
            bi_7110_dt_1030_solido_lamb = df_tot['bi_7110_dt_1030_solido_lamb'].iloc[-1]
            bi_7110_dt_1030_solido_q    = df_tot['bi_7110_dt_1030_solido_q'].iloc[-1]
            bi_7110_dt_1030_solido_sp   = df_tot['bi_7110_dt_1030_solido_sp'].iloc[-1]
            bi_7110_dt_1030_solido_sup  = df_tot['bi_7110_dt_1030_solido_sup'].iloc[-1]
            bi_7110_dt_1030_solido_t    = df_tot['bi_7110_dt_1030_solido_t'].iloc[-1]
            bk_7110_ft_1030_dinf        = df_tot['bk_7110_ft_1030_dinf'].iloc[-1]
            bk_7110_ft_1030_dsup        = df_tot['bk_7110_ft_1030_dsup'].iloc[-1]
            bk_7110_ft_1030_inf         = df_tot['bk_7110_ft_1030_inf'].iloc[-1]
            bk_7110_ft_1030_r           = df_tot['bk_7110_ft_1030_r'].iloc[-1]
            bk_7110_ft_1030_sup         = df_tot['bk_7110_ft_1030_sup'].iloc[-1]
            bo_7110_lt_1009_s4_inf      = df_tot['bo_7110_lt_1009_s4_inf'].iloc[-1]
            bo_7110_lt_1009_s4_lamb     = df_tot['bo_7110_lt_1009_s4_lamb'].iloc[-1]
            bo_7110_lt_1009_s4_q        = df_tot['bo_7110_lt_1009_s4_q'].iloc[-1]
            bo_7110_lt_1009_s4_sp       = df_tot['bo_7110_lt_1009_s4_sp'].iloc[-1]
            bo_7110_lt_1009_s4_sup      = df_tot['bo_7110_lt_1009_s4_sup'].iloc[-1]
            bo_7110_lt_1009_s4_t        = df_tot['bo_7110_lt_1009_s4_t'].iloc[-1]
            bp_7110_ot_1003_inf         = df_tot['bp_7110_ot_1003_inf'].iloc[-1]
            bp_7110_ot_1003_lamb        = df_tot['bp_7110_ot_1003_lamb'].iloc[-1]
            bp_7110_ot_1003_q           = df_tot['bp_7110_ot_1003_q'].iloc[-1]
            bp_7110_ot_1003_sp          = df_tot['bp_7110_ot_1003_sp'].iloc[-1]
            bp_7110_ot_1003_sup         = df_tot['bp_7110_ot_1003_sup'].iloc[-1]
            bp_7110_ot_1003_t           = df_tot['bp_7110_ot_1003_t'].iloc[-1]
            bq_7110_pt_1010_inf         = df_tot['bq_7110_pt_1010_inf'].iloc[-1]
            bq_7110_pt_1010_lamb        = df_tot['bq_7110_pt_1010_lamb'].iloc[-1]
            bq_7110_pt_1010_sp           = df_tot['bq_7110_pt_1010_sp'].iloc[-1]
            bq_7110_pt_1010_q           = df_tot['bq_7110_pt_1010_q'].iloc[-1]
            bq_7110_pt_1010_sup         = df_tot['bq_7110_pt_1010_sup'].iloc[-1]
            bq_7110_pt_1010_t           = df_tot['bq_7110_pt_1010_t'].iloc[-1]
            br_7120_ft_1002_dinf        = df_tot['br_7120_ft_1002_dinf'].iloc[-1]
            br_7120_ft_1002_dsup        = df_tot['br_7120_ft_1002_dsup'].iloc[-1]
            br_7120_ft_1002_inf         = df_tot['br_7120_ft_1002_inf'].iloc[-1]
            br_7120_ft_1002_r           = df_tot['br_7120_ft_1002_r'].iloc[-1]
            br_7120_ft_1002_sup         = df_tot['br_7120_ft_1002_sup'].iloc[-1]
		  
		# Valor que ajusta el número de decimales mostrados en la pantalla            
        decimal = 2
       

		# Se retornan los valores redondeados en formato json 
		
        return jsonify(datetime=date_time, bi_7110_dt_1030_solido_inf = round( bi_7110_dt_1030_solido_inf , decimal), bi_7110_dt_1030_solido_lamb = round( bi_7110_dt_1030_solido_lamb , decimal),
										bi_7110_dt_1030_solido_q = round( bi_7110_dt_1030_solido_q , decimal),
										    bi_7110_dt_1030_solido_sp= round( bi_7110_dt_1030_solido_sp , decimal),  bi_7110_dt_1030_solido_sup= round( bi_7110_dt_1030_solido_sup , decimal), bi_7110_dt_1030_solido_t =  round( bi_7110_dt_1030_solido_t , decimal),
										    bk_7110_ft_1030_dinf= round( bk_7110_ft_1030_dinf , decimal), bk_7110_ft_1030_dsup = round( bk_7110_ft_1030_dsup , decimal), bk_7110_ft_1030_inf = round( bk_7110_ft_1030_inf , decimal),
										    bk_7110_ft_1030_r= round( bk_7110_ft_1030_r , decimal), bk_7110_ft_1030_sup = round( bk_7110_ft_1030_sup , decimal), bo_7110_lt_1009_s4_inf = round( bo_7110_lt_1009_s4_inf , decimal),
										   bo_7110_lt_1009_s4_lamb = round( bo_7110_lt_1009_s4_lamb , decimal), bo_7110_lt_1009_s4_q = round( bo_7110_lt_1009_s4_q , decimal), bo_7110_lt_1009_s4_sp  = round( bo_7110_lt_1009_s4_sp , decimal),
										    bo_7110_lt_1009_s4_sup= round( bo_7110_lt_1009_s4_sup , decimal), bo_7110_lt_1009_s4_t= round( bo_7110_lt_1009_s4_t , decimal), bp_7110_ot_1003_inf = round( bp_7110_ot_1003_inf , decimal), 
										   bp_7110_ot_1003_lamb = round( bp_7110_ot_1003_lamb , decimal), bp_7110_ot_1003_q= round( bp_7110_ot_1003_q , decimal),bp_7110_ot_1003_sp = round(  bp_7110_ot_1003_sp , decimal), 
										    bp_7110_ot_1003_sup= round( bp_7110_ot_1003_sup , decimal), bp_7110_ot_1003_t= round( bp_7110_ot_1003_t , decimal), bq_7110_pt_1010_inf = round( bq_7110_pt_1010_inf , decimal), 
										   bq_7110_pt_1010_lamb = round( bq_7110_pt_1010_lamb , decimal), bq_7110_pt_1010_q= round( bq_7110_pt_1010_q , decimal), bq_7110_pt_1010_sp = round( bq_7110_pt_1010_sp , decimal), 
										    bq_7110_pt_1010_sup= round( bq_7110_pt_1010_sup , decimal),bq_7110_pt_1010_t = round( bq_7110_pt_1010_t , decimal), br_7120_ft_1002_dinf = round( br_7120_ft_1002_dinf , decimal), 
										    br_7120_ft_1002_dsup= round( br_7120_ft_1002_dsup , decimal),br_7120_ft_1002_inf = round( br_7120_ft_1002_inf , decimal), br_7120_ft_1002_r = round( br_7120_ft_1002_r , decimal), 
										   br_7120_ft_1002_sup = round( br_7120_ft_1002_sup , decimal))

    except Exception as e:
     return str(e)

@app.route('/_get_dust_forecast', methods=['POST'])
def _get_dust_forecast():

# Esta función obtiene los valores de la predicción de "Probabilidad de polvo". Con 1, 2 y 3 días hacia adelante, se retornan tres valores 0,1 o 2 si la probabilidad es "Baja, "Media" o "alta, respectivamente".
# Los valores se obtienen con peticiones a la tabla "from fondef.Clasificador_dust_classification", donde Clasificador corresponde al nombre del clasificador, por ejemplo: svm.
    try:

        if request.method == 'POST':
			
			# Se obtiene el nombre del clasificador
            Clasificador = request.values['Clasificador']
            
            # Se generan los nombres de los tags en la tabla cassandra
            DustTags = [ Clasificador +"_1_ahead"   , Clasificador +"_2_ahead"  ,  Clasificador +"_3_ahead" ]
            
            query = "select * from fondef." + Clasificador + "_dust_classification where sensor = '{}' LIMIT 1"

			# Variable que guardará los datos de las predicciones            
            df_tot = [0] * len(DustTags)

			# Peticiones del valor más reciente de cada predicción a la tabla de "fondef.Clasificador_dust_classification" de cassandra
            for r in range(0, len(DustTags)):
                rows = connection.execute(
                    query.format(DustTags[r]))
                df = rows._current_rows
                df = df.iloc[::-1]
                df = df.reset_index(drop=True)
                df_tot[r] = df

            df_tot = pd.concat(df_tot, ignore_index=True)
            df_tot = pd.pivot_table(df_tot, values='value', columns='sensor', index='datetime')
            df_tot = df_tot.reset_index(drop=False)

            df_tot['datetime'] = df_tot['datetime'].dt.tz_localize('utc').dt.tz_convert('America/Caracas')
            df_tot['datetime'] = df_tot['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
			
			# Variable que guarda la fecha y hora de los datos
            datetime = df_tot["datetime"].iloc[-1]
            df_tot = df_tot.set_index('datetime', drop=True)
            df_tot = df_tot.reset_index(drop=True)

			# Asignación de variables a cada valor de cada predicción 
            Prob_1_ahead = df_tot[ DustTags[0] ].iloc[-1] # 1 día hacia adelante
            Prob_2_ahead = df_tot[ DustTags[1] ].iloc[-1] # 2 día hacia adelante
            Prob_3_ahead = df_tot[ DustTags[2] ].iloc[-1] # 3 día hacia adelante

		# Se retornan los valores en formato json 
        return jsonify(datetime=datetime, Prob_1_ahead=Prob_1_ahead,
						Prob_2_ahead=Prob_2_ahead,Prob_3_ahead=Prob_3_ahead)
						
    except Exception as e:
     return str(e)

@app.route('/_send_control_value', methods=['POST'])
def send_control_value():

# Esta función guarda los valores ingresados para los parámetros del controlador por el usuario en la página web en la tabla "fondef.mpc_weights".

    try:
					
	 # Se obtiene el nombre del parámetro
     param = request.form['param_id']
		
	 # Se obtiene el valor del parámetro a guardar
     value = float(request.form['value'])
     
     # Variable que guarda la fecha y hora de los datos
     date_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

	 # Se guarda el valor y el tiempo de escritura asociados al parámetros

     connection.execute(
			"""INSERT INTO fondef.mpc_weights (sensor, datetime, value)
			VALUES (%(sensor)s, %(datetime)s, %(value)s)""",
			{'sensor': param, 'datetime': date_time, 'value': value}
		)

	 # Variable que guardará los datos de los parámetros
     df_tot = [0] * len(todosTags_control)
			
    

	# Peticiones del valor más reciente de cada sensor a la tabla de "fondef.mpc_weights" de cassandra
     for r in range(0, len(todosTags_control)):

      rows = connection.execute(
			"select * from fondef.mpc_weights where sensor = '{}' LIMIT 1".format(todosTags_control[r]))
      df = rows._current_rows
      df = df.iloc[::-1]
      df = df.reset_index(drop=True)
      df_tot[r] = df
					
     df_tot = pd.concat(df_tot, ignore_index=True)
     df_tot = pd.pivot_table(df_tot, values='value', columns='sensor')
     df_tot = df_tot.reset_index(drop=False)
     
     
	# Peticiones del valor más reciente de cada sensor a la tabla de "fondef.mpc_weights" de cassandra
     for r in range(0, len(todosTags_control)):
      connection.execute(
			"""INSERT INTO fondef.mpc_weights (sensor, datetime, value)
			VALUES (%(sensor)s, %(datetime)s, %(value)s)""",
			{'sensor': todosTags_control[r], 'datetime': date_time, 'value': df_tot[todosTags_control[r]].iloc[-1]}
		) 
     

     return jsonify(resp="Exito")
    
    except Exception as e:
        return jsonify(resp="Fracaso")

####################################################################################
##################################  Gráficos de la pestaña de tendencias ######################################
####################################################################################

	#################### Gráfico de Flujo de descarga ################################ 
	################################################################################## 

@app.route('/make_ajax_plot_1_1', methods=['POST'])
def make_ajax_plot_1_1(Escala,resample):
	
# Esta función genera un gráfico de la variable "Flujo de descarga"
       
    # Se ajustan las funcionalidades que tendrá el gráfico
    TOOLTIPS = [
            ("fecha", "@x{%F}"),
            ("hora", "@x{%R}"),
            ("valor", "@y{0.0}"),
        ]    
    TOOLS="pan,wheel_zoom,box_zoom,reset,save"            
    
    # Se crea la variable que contiene la información del gráfico    
    plot = figure(plot_height=450, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)
  
    
    ###########################################################
    ## Obtención de los datos de predicciones pasadas

	# Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)
    source1 = AjaxDataSource(data_url=request.url_root + 'get_source_past_forecast_controller_b_ufr_optimal/', polling_interval=int(10000), mode='append' )
    
    # Se obtienen los datos de las predicciones pasadas
    df2 = get_source_past_forecast_controller(Escala , "b_ufr_optimal",resample)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source1.data = dict(x=Dates, y=Values) 

	# Se asigna una curva y puntos a los valores obtenidos
    r1 = plot.line(x='x' , y='y', source=source1, line_width=2, color = 'green')
    r2 = plot.circle(x='x' , y='y', size=5, source=source1, color="green", fill_alpha=1) 
    
    ###########################################################
    ## Obtención de las predicciones que se generan de la variable
    
    # Variable que guarda los datos de la predicción y se actualiza completamente cada 10 segundos
    source3 = AjaxDataSource(data_url=request.url_root + 'data_forecast_controller_b_ufr_optimal/', polling_interval=int(10000), mode='replace' , max_size = 18 )
	
	# Se obtienen los datos de las predicciones
    df2 = get_source_forecast_controller(Escala , "b_ufr_optimal")
    
    # Fecha de la predicción
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Variable que se usará para asignar una fecha y hora para cada punto a graficar
    t = []
    
    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos
    for i in range( len(Values[0]) ):
     t.append(		pd.to_datetime(Dates[0] + np.timedelta64(i*5,'m'))		)
    
    # Se guardan las fechas y valores
    source3.data = dict(x=t, y=Values[0]) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r3 = plot.line(x='x', y='y', source=source3, line_width=2, color = 'green')
    r4 = plot.circle(x='x', y='y', size=5, source=source3, color="green", fill_alpha=1)
    
    ###########################################################
    ## Obtención de los valores pasados obtenidos de los sensores
    
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)
    source4 = AjaxDataSource(data_url=request.url_root + 'data_past_sensor_bk_7110_ft_1030/', polling_interval=int(10000), mode='append' )

	 # Se obtienen los datos pasados de los sensores
    df2 = get_source_sensor(Escala , "bk_7110_ft_1030",resample,0)

	# Fecha de cada dato a graficar
	
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source4.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r5 = plot.line(x='x', y='y', source=source4, line_width=2, color = 'red')
    r6 = plot.circle(x='x', y='y', size=5, source=source4, color="red", fill_alpha=1)
	###########################################################
    ## Se ajustan los parámetros que ajustan la información de los ejes del gráfico

    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter(seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"x": 'datetime'} ))     
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"

    # Se define la leyenda del gráfico
    legend = Legend(items=[
    ("Flujo de descarga predicho"   , [r1, r2,r3,r4]),
    ("Flujo de descarga real" , [r5,r6]),
     ], label_text_font= "bookman",location=(0, 0),orientation = "horizontal",click_policy = "hide")
    plot.add_layout(legend, 'below')

    script, div = components(plot)
    return script, div

@app.route('/get_source_past_forecast_controller_b_ufr_optimal/', methods=['POST'])
def get_source_past_forecast_controller_b_ufr_optimal():
 
# Esta función obtiene el último valor pasado de la predicción de la variable "Flujo de descarga"
 
	# Se obtienen los datos pasados de las predicciones
    df2 = get_source_past_forecast_controller(2 , "b_ufr_optimal",5/60)
    
    # Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1])
    
    # Valor del último dato
    Dato = Values[-1]
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)    

@app.route('/data_forecast_controller_b_ufr_optimal/', methods=['POST'])
def data_forecast_controller_b_ufr_optimal():
	
	# Esta función obtiene la última predicción de la variable "Flujo de descarga"
	
	# Se obtienen los datos de la predicción
    df2 = get_source_forecast_controller(2 , "b_ufr_optimal")
    
    # Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Variable que se usará para asignar una fecha y hora para cada punto a graficar
    t = []
    
    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos
    for i in range(18):
     t.append(	datetime.datetime.now(tz = pytz.timezone('America/Caracas')) + datetime.timedelta(minutes=i*5))
    
    # Valores de la última predicción
    Dato = Values[0] #
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=t, y=Dato)

    return  serialize_json(data)

@app.route('/data_past_sensor_bk_7110_ft_1030/', methods=['POST'])
def data_past_sensor_bk_7110_ft_1030():
	
# Esta función obtiene el último valor pasado de la variable "Flujo de descarga"
 
	# Se obtienen los datos de los valores pasado
    df2 = get_source_sensor(2 , "bk_7110_ft_1030",0,0)
    
    # Fecha de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1])
    
    # Valor del último dato
    Dato = Values[-1] 
      
    # Se guardan la fecha y el valor del dato
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)
    
	#################### Gráfico de Flujo Floculante ################################ 
	##################################################################################     

@app.route('/make_ajax_plot_1_2', methods=['POST'])
def make_ajax_plot_1_2(Escala,resample):
	
# Esta función genera un gráfico de la variable "Flujo de floculante"	
	
	# Se ajustan las funcionalidades que tendrá el gráfico
    TOOLTIPS = [
            ("fecha", "@x{%F}"),
            ("hora", "@x{%R}"),
            ("valor", "@y{0.0}"),
        ]        
    TOOLS="pan,wheel_zoom,box_zoom,reset,save"                
    
    # Se crea la variable que contiene la información del gráfico    
    plot = figure(plot_height=450, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)
    
    ###########################################################
    ## Obtención de los datos de predicciones pasadas
        
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)    
    source1 = AjaxDataSource(data_url=request.url_root + 'get_source_past_forecast_controller_c_ffr_optimal/', polling_interval=int(10000), mode='append' )
    
    # Se obtienen los datos de las predicciones pasadas
    df2 = get_source_past_forecast_controller(Escala , "c_ffr_optimal",resample)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar    
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source1.data = dict(x=Dates, y=Values) 

	# Se asigna una curva y puntos a los valores obtenidos
    r1 = plot.line(x='x' , y='y', source=source1, line_width=2, color = 'green')
    r2 = plot.circle(x='x' , y='y', size=5, source=source1, color="green", fill_alpha=1) 
    
    ###########################################################
    ## Obtención de las predicciones que se generan de la variable
    
    # Variable que guarda los datos de la predicción y se actualiza completamente cada 10 segundos
    source3 = AjaxDataSource(data_url=request.url_root + 'data_forecast_controller_c_ffr_optimal/', polling_interval=int(10000), mode='replace' , max_size = 18 )
	
	# Se obtienen los datos de las predicciones
    df2 = get_source_forecast_controller(Escala , "c_ffr_optimal")
    
    # Fecha de la predicción
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Variable que se usará para asignar una fecha y hora para cada punto a graficar
    t = []
    
    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos
    for i in range(len( Values[0] )):
     t.append(		pd.to_datetime(Dates[0] + np.timedelta64(i*5,'m'))		)
    
    # Se guardan las fechas y valores
    source3.data = dict(x=t, y=Values[0]) 
	
	# Se asigna una curva y puntos a los valores obtenidos
    r3 = plot.line('x', 'y', source=source3, line_width=2, color = 'green')
    r4 = plot.circle('x','y', size=5, source=source3, color="green", fill_alpha=1)

    ###########################################################
    ## Obtención de los valores pasados obtenidos de los sensores
    
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)
    source4 = AjaxDataSource(data_url=request.url_root + 'data_past_sensor_br_7120_ft_1002/', polling_interval=int(10000), mode='append' )

	# Se obtienen los datos pasados de los sensores
    df2 = get_source_sensor(Escala , "br_7120_ft_1002",resample,0)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source4.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r5 = plot.line('x', 'y', source=source4, line_width=2, color = 'red')
    r6 = plot.circle('x', 'y', size=5, source=source4, color="red", fill_alpha=1)
	###########################################################	
    ## Se ajustan los parámetros que ajustan la información de los ejes del gráfico
    
    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter(seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"x": 'datetime'} ))     
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"    

    # Se define la leyenda del gráfico
    legend = Legend(items=[
    ("Flujo Floculante predicho"   , [r1, r2,r3,r4]),
    ("Flujo Floculante real" , [r5,r6]),
     ], label_text_font= "bookman",location=(0, 0),orientation = "horizontal",click_policy = "hide")
    plot.add_layout(legend, 'below')    
    
    script, div = components(plot)
    return script, div

@app.route('/get_source_past_forecast_controller_c_ffr_optimal/', methods=['POST'])
def get_source_past_forecast_controller_c_ffr_optimal():
 
# Esta función obtiene el último valor pasado de la predicción de la variable "Flujo de floculante"
 
	# Se obtienen los datos pasados de las predicciones
    df2 = get_source_past_forecast_controller(2 , "c_ffr_optimal",5/60)
    
    # Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1]) #
    
    # Valor del último dato
    Dato = Values[-1] #
      
    # Se guardan la fecha y el valor del dato  
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)    

@app.route('/data_forecast_controller_c_ffr_optimal/', methods=['POST'])
def data_forecast_controller_c_ffr_optimal():
	
# Esta función obtiene la última predicción de la variable "Flujo de floculante"
    
    # Se obtienen los datos de la predicción
    df2 = get_source_forecast_controller(2 , "c_ffr_optimal")
    
    
    # Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Variable que se usará para asignar una fecha y hora para cada punto a graficar
    t = []
    
    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos
    for i in range(18):
     t.append(	datetime.datetime.now(tz = pytz.timezone('America/Caracas')) + datetime.timedelta(minutes=i*5))
    
    # Valores de la última predicción
    Dato = Values[0] 
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=t, y=Dato)
    
    return  serialize_json(data)
                        
@app.route('/data_past_sensor_br_7120_ft_1002/', methods=['POST'])
def data_past_sensor_br_7120_ft_1002():
 
# Esta función obtiene el último valor pasado de la variable "Flujo de floculante"

	# Se obtienen los datos de los valores pasado
    df2 = get_source_sensor(2 , "br_7120_ft_1002",0,0)
    
    # Fecha de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1]) #
    
    # Valor del último dato
    Dato = Values[-1] #
    
    # Se guardan la fecha y el valor del dato  
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

	#################### Gráfico de Setpoint bomba 3 ################################ 
	##################################################################################

@app.route('/make_ajax_plot_1_3', methods=['POST'])
def make_ajax_plot_1_3(Escala,resample):
	
# Esta función genera un gráfico de la variable "Setpoint bomba 3"
	
	# Se ajustan las funcionalidades que tendrá el gráfico
    
    TOOLTIPS = [
            ("fecha", "@x{%F}"),
            ("hora", "@x{%R}"),
            ("valor", "@y{0.0}"),
        ]      

    TOOLS="pan,wheel_zoom,box_zoom,reset,save"                
    
    # Se crea la variable que contiene la información del gráfico    
    plot = figure(plot_height=450, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)
        
    ###########################################################
    ## Obtención de los datos del setpoint del controlador
    
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)
    source1 = AjaxDataSource(data_url=request.url_root + 'get_source_sensor_forecast_controller_cc_4310_pu_0003_sp_ext/', polling_interval=int(10000), mode='append' )

    # Se obtienen los datos
    df2 = get_source_sensor(Escala , "cc_4310_pu_0003_sp_ext",resample,0)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source1.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r1 = plot.line('x', 'y', source=source1, line_width=2, color = 'green')
    r2 = plot.circle('x','y', size=5, source=source1, color="green", fill_alpha=1)
    
    ###########################################################
    ## Obtención de los datos del setpoint real (lo que realmente se midió)
        
	# Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)        
    source4 = AjaxDataSource(data_url=request.url_root + 'data_past_sensor_cg_0003_pv/', polling_interval=int(10000), mode='append' )

	# Se obtienen los datos pasados de los sensores
    df2 = get_source_sensor(Escala , "cg_0003_pv",resample,0)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source4.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r3 = plot.line('x', 'y', source=source4, line_width=2, color = 'red')
    r4 = plot.circle('x', 'y', size=5, source=source4, color="red", fill_alpha=1)
	###########################################################
	## Se ajustan los parámetros que ajustan la información de los ejes del gráfico
	
    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter(seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"x": 'datetime'} ))     
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"    

	# Se define la leyenda del gráfico
    legend = Legend(items=[
    ("Setpoint bomba 3 controlador"   , [r1, r2]),
    ("Setpoint bomba 3 real" , [r3,r4]),
     ], label_text_font= "bookman",location=(0, 0),orientation = "horizontal",click_policy = "hide")
    plot.add_layout(legend, 'below')        
    
    script, div = components(plot)
    return script, div
    
@app.route('/get_source_sensor_forecast_controller_cc_4310_pu_0003_sp_ext/', methods=['POST'])
def get_source_sensor_forecast_controller_cc_4310_pu_0003_sp_ext():

# Obtención del último dato del "setpoint bomba 3" del controlador 

	# Se obtienen los datos pasados de las predicciones
    df2 = get_source_sensor(2 , "cc_4310_pu_0003_sp_ext",0,0)
    
    # Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1]) #
    
    # Valor del último dato
    Dato = Values[-1] #
    
    # Se guardan la fecha y el valor del dato  
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)
    
@app.route('/data_past_sensor_cg_0003_pv/', methods=['POST'])
def data_past_sensor_cg_0003_pv():
 
# Esta función obtiene el último valor pasado de la variable "setpoint bomba 3"

	# Se obtienen los datos de los valores pasado
    df2 = get_source_sensor(2 , "cg_0003_pv",0,0)
    
    # Fecha de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1])
    
    # Valor del último dato
    Dato = Values[-1]
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=[t], y=[Dato])

    return  serialize_json(data)
        
	#################### Gráfico de Setpoint bomba 4 ################################ 
	##################################################################################

@app.route('/make_ajax_plot_1_4', methods=['POST'])
def make_ajax_plot_1_4(Escala,resample):
	
# Esta función genera un gráfico de la variable "Setpoint bomba 4"
	
	# Se ajustan las funcionalidades que tendrá el gráfico
    
    TOOLTIPS = [
            ("fecha", "@x{%F}"),
            ("hora", "@x{%R}"),
            ("valor", "@y{0.0}"),
        ]      	
        
    TOOLS="pan,wheel_zoom,box_zoom,reset,save"   
    
    # Se crea la variable que contiene la información del gráfico                 
    plot = figure(plot_height=450, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)

    ###########################################################
    ## Obtención de los datos del setpoint del controlador
    
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)
    source1 = AjaxDataSource(data_url=request.url_root + 'get_source_sensor_controller_cd_4310_pu_0004_sp_ext/', polling_interval=int(10000), mode='append' )
	
	# Se obtienen los datos
    df2 = get_source_sensor(Escala , "cd_4310_pu_0004_sp_ext",resample,0)
	
	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source1.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r1 = plot.line('x', 'y', source=source1, line_width=2, color = 'green')
    r2 = plot.circle('x','y', size=5, source=source1, color="green", fill_alpha=1)
    
    ###########################################################
    ## Obtención de los datos del setpoint real (lo que realmente se midió)
        
	# Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)        
    source4 = AjaxDataSource(data_url=request.url_root + 'data_past_sensor_ch_0004_pv/', polling_interval=int(10000), mode='append' )

	# Se obtienen los datos pasados de los sensores
    df2 = get_source_sensor(Escala , "ch_0004_pv",resample,0)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source4.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r3 = plot.line('x', 'y', source=source4, line_width=2, color = 'red')
    r4 = plot.circle('x', 'y', size=5, source=source4, color="red", fill_alpha=1)
	###########################################################
	## Se ajustan los parámetros que ajustan la información de los ejes del gráfico

    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter(seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"x": 'datetime'} ))     
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"

	# Se define la leyenda del gráfico
    legend = Legend(items=[
    ("Setpoint bomba 4 controlador"   , [r1, r2]),
    ("Setpoint bomba 4 real" , [r3,r4]),
     ],label_text_font= "bookman", location=(0, 0),orientation = "horizontal",click_policy = "hide")
    plot.add_layout(legend, 'below')   
    
    script, div = components(plot)
    return script, div
    
@app.route('/get_source_sensor_controller_cd_4310_pu_0004_sp_ext/', methods=['POST'])
def get_source_sensor_controller_cd_4310_pu_0004_sp_ext():

## Obtención del último dato del "setpoint bomba 4" del controlador

	# Se obtienen los datos pasados de las predicciones
    df2 = get_source_sensor(2 , "cd_4310_pu_0004_sp_ext",0,0)
    
    # Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1])
    
    # Valor del último dato
    Dato = Values[-1]

	# Se guardan la fecha y el valor del dato        
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)
    
@app.route('/data_past_sensor_ch_0004_pv/', methods=['POST'])
def data_past_sensor_ch_0004_pv():

# Esta función obtiene el último valor pasado de la variable "setpoint bomba 4"

	# Se obtienen los datos de los valores pasado
    df2 = get_source_sensor(2 , "ch_0004_pv",0,0)
    
    # Fecha de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1])
    
    # Valor del último dato
    Dato = Values[-1]
      
    # Se guardan la fecha y el valor del dato
    data = dict(x=[t], y=[Dato])

    return  serialize_json(data)

	#################### Gráfico de Presión hidrostática ################################ 
	##################################################################################

@app.route('/make_ajax_plot_1_5', methods=['POST'])
def make_ajax_plot_1_5(Escala,resample):
	
# Esta función genera un gráfico de la variable "Presión hidrostática"
	
	# Se ajustan las funcionalidades que tendrá el gráfico
    TOOLTIPS = [
            ("fecha", "@x{%F}"),
            ("hora", "@x{%R}"),
            ("valor", "@y{0.0}"),
        ]      
        	
    TOOLS="pan,wheel_zoom,box_zoom,reset,save"                
    
    # Se crea la variable que contiene la información del gráfico
    plot = figure(plot_height=450, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)
    
    ###########################################################
    ## Obtención de los datos de predicciones pasadas
    
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)
    source1 = AjaxDataSource(data_url=request.url_root + 'get_source_past_forecast_controller_d_pressure_optimal/', polling_interval=int(10000), mode='append' )
    
    # Se obtienen los datos de las predicciones pasadas
    df2 = get_source_past_forecast_controller(Escala , "d_pressure_optimal",resample)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source1.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r3 = plot.line('x', 'y', source=source1, line_width=2, color = 'green')
    r4 = plot.circle('x','y', size=5, source=source1, color="green", fill_alpha=1)
        
    ###########################################################
    ## Obtención de las predicciones que se generan de la variable

	# Variable que guarda los datos de la predicción y se actualiza completamente cada 10 segundos
    source1 = AjaxDataSource(data_url=request.url_root + 'data_forecast_controller_d_pressure_optimal/', polling_interval=int(10000), mode='replace' , max_size = 18 )
	
	# Se obtienen los datos de las predicciones
    df2 = get_source_forecast_controller(Escala , "d_pressure_optimal")
    
    # Fecha de la predicción
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Variable que se usará para asignar una fecha y hora para cada punto a graficar
    t = []
    
    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos
    for i in range(len( Values[0] )):
     t.append(		pd.to_datetime(Dates[0] + np.timedelta64(i*5,'m'))		)
    
	# Se guardan las fechas y valores
    source1.data = dict(x=t, y=Values[0]) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r1 = plot.line('x', 'y', source=source1, line_width=2, color = 'green')
    r2 = plot.circle('x','y', size=5, source=source1, color="green", fill_alpha=1)
    ###########################################################
    ## Obtención de los valores pasados obtenidos de los sensores

    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)
    source4 = AjaxDataSource(data_url=request.url_root + 'data_past_sensor_bq_7110_pt_1010/', polling_interval=int(10000), mode='append')

	# Se obtienen los datos pasados de los sensores
    df2 = get_source_sensor(Escala , "bq_7110_pt_1010",resample,0)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source4.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r5 = plot.line(x='x', y='y', source=source4, line_width=2, color = 'red')
    r6 = plot.circle(x='x', y='y', size=5, source=source4, color="red", fill_alpha=1)
	###########################################################
	## Se ajustan los parámetros que ajustan la información de los ejes del gráfico
	
    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter(seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"x": 'datetime'} ))
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"

	# Se define la leyenda del gráfico
    legend = Legend(items=[
    ("Presión hidrostática predicha"   , [r1, r2,r3,r4]),
    ("Presión hidrostática real" , [r5,r6]),
     ], label_text_font= "bookman",location=(0, 0),orientation = "horizontal",click_policy = "hide")
    plot.add_layout(legend, 'below')   
    
    script, div = components(plot)
    return script, div

@app.route('/get_source_past_forecast_controller_d_pressure_optimal/', methods=['POST'])
def get_source_past_forecast_controller_d_pressure_optimal():
 
# Esta función obtiene el último valor pasado de la predicción de la variable "Presión hidrostática"

	# Se obtienen los datos pasados de las predicciones
    df2 = get_source_past_forecast_controller(2 , "d_pressure_optimal",5/60)
    
    # Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
	
	# Fecha del último dato
    t = pd.to_datetime(Dates[-1])
    
    # Valor del último dato
    Dato = Values[-1]
      
    # Se guardan la fecha y el valor del dato
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_forecast_controller_d_pressure_optimal/', methods=['POST'])
def data_forecast_controller_d_pressure_optimal():

# Esta función obtiene la última predicción de la variable "Presión hidrostática"

	# Se obtienen los datos de la predicción
    df2 = get_source_forecast_controller(2 , "d_pressure_optimal")
		
	# Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Variable que se usará para asignar una fecha y hora para cada punto a graficar
    t = []
    
    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos
    for i in range(18):
     t.append(	datetime.datetime.now(tz = pytz.timezone('America/Caracas')) + datetime.timedelta(minutes=i*5))
    
    # Valores de la última predicción
    Dato = Values[0]
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=t, y=Dato)

    return  serialize_json(data)

@app.route('/data_past_sensor_bq_7110_pt_1010/', methods=['POST'])
def data_past_sensor_bq_7110_pt_1010():
 
# Esta función obtiene el último valor pasado de la variable "Presión hidrostática"

	# Se obtienen los datos de los valores pasado
    df2 = get_source_sensor(2 , "bq_7110_pt_1010",0,0)
    
    # Fecha de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1])
    
    # Valor del último dato
    Dato = Values[-1]
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=[t], y=[Dato])

    return  serialize_json(data)
    
	#################### Gráfico de Concentración sólidos descarga ################################ 
	##################################################################################

@app.route('/make_ajax_plot_1_6', methods=['POST'])
def make_ajax_plot_1_6(Escala,resample):
	
# Esta función genera un gráfico de la variable "Concentración sólidos descarga"	
	
    TOOLTIPS = [
            ("fecha", "@x{%F}"),
            ("hora", "@x{%R}"),
            ("valor", "@y{0.0}"),
        ]      	
    
    TOOLS="pan,wheel_zoom,box_zoom,reset,save"        
    
    # Se crea la variable que contiene la información del gráfico         
    plot = figure(plot_height=470, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)
	###########################################################
    ## Obtención de los datos de predicciones pasadas
    
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)
    source1 = AjaxDataSource(data_url=request.url_root + 'get_source_past_forecast_controller_e_usc_optimal/', polling_interval=int(10000), mode='append' )
    
    # Se obtienen los datos de las predicciones pasadas
    df2 = get_source_past_forecast_controller(Escala , "e_usc_optimal",resample)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source1.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r3 = plot.line('x', 'y', source=source1, line_width=2, color = 'green')
    r4 = plot.circle('x','y', size=5, source=source1, color="green", fill_alpha=1)

    ###########################################################
    ## Obtención de las predicciones que se generan de la variable
    
	# Variable que guarda los datos de la predicción y se actualiza completamente cada 10 segundos
    source2 = AjaxDataSource(data_url=request.url_root + 'data_forecast_controller_e_usc_optimal/', polling_interval=int(10000), mode='replace' , max_size = 18 )
	
	# Se obtienen los datos de las predicciones
    df2 = get_source_forecast_controller(Escala , "e_usc_optimal")
    
    # Fecha de la predicción
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Variable que se usará para asignar una fecha y hora para cada punto a graficar
    t = []
    
    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos
    for i in range(len( Values[0] )):
     t.append(		pd.to_datetime(Dates[0] + np.timedelta64(i*5,'m'))		)
    
    # Se guardan las fechas y valores
    source2.data = dict(x=t, y=Values[0]) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r1 = plot.line('x', 'y', source=source2, line_width=2, color = 'green')
    r2 = plot.circle('x','y', size=5, source=source2, color="green", fill_alpha=1)

    ###########################################################
    ## Obtención de los valores pasados obtenidos de los sensores
    
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)
    source4 = AjaxDataSource(data_url=request.url_root + 'data_past_sensor_bi_7110_dt_1030_solido/', polling_interval=int(10000), mode='append' )

	# Se obtienen los datos pasados de los sensores
    df2 = get_source_sensor(Escala , "bi_7110_dt_1030_solido",resample,0)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source4.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r5 = plot.line(x='x', y='y', source=source4, line_width=2, color = 'red')
    r6 = plot.circle(x='x', y='y', size=5, source=source4, color="red", fill_alpha=1)
    
	###########################################################
    ## Se ajustan los parámetros que ajustan la información de los ejes del gráfico
    
    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter(seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"x": 'datetime'} ))     
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"

    # Se define la leyenda del gráfico
    legend = Legend(items=[
    ("Concentración sólidos descarga predicha"   , [r1, r2,r3,r4]),
    ("Concentración sólidos descarga real" , [r5,r6]),
     ], label_text_font= "bookman",location=(0, 0),orientation = "vertical",click_policy = "hide")
    plot.add_layout(legend, 'below')

    script, div = components(plot)
    return script, div

@app.route('/get_source_past_forecast_controller_e_usc_optimal/', methods=['POST'])
def get_source_past_forecast_controller_e_usc_optimal():
 
# Esta función obtiene el último valor pasado de la predicción de la variable "Concentración sólidos descarga"

	# Se obtienen los datos pasados de las predicciones
    df2 = get_source_past_forecast_controller(2 , "e_usc_optimal",5/60)
    
    # Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1])
	
	# Valor del último dato    
    Dato = Values[-1]
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=[t], y=[Dato])

    return  serialize_json(data)

@app.route('/data_forecast_controller_e_usc_optimal/', methods=['POST'])
def data_forecast_controller_e_usc_optimal():
	
# Esta función obtiene la última predicción de la variable "Concentración sólidos descarga"
	
	# Se obtienen los datos de la predicción
    df2 = get_source_forecast_controller(2 , "e_usc_optimal")
    
    # Fechas de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Variable que se usará para asignar una fecha y hora para cada punto a graficar
    t = []
    
    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos
    for i in range(len( Values[0] )):
     t.append(	datetime.datetime.now(tz = pytz.timezone('America/Caracas')) + datetime.timedelta(minutes=i*5))
    
    # Valores de la última predicción
    Dato = Values[0] 
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=t, y=Dato)

    return  serialize_json(data)

@app.route('/data_past_sensor_bi_7110_dt_1030_solido/', methods=['POST'])
def data_past_sensor_bi_7110_dt_1030_solido():
 
# Esta función obtiene el último valor pasado de la variable "Concentración sólidos descarga"
 
	# Se obtienen los datos de los valores pasado
    df2 = get_source_sensor(2 , "bi_7110_dt_1030_solido",0,0)
    
    # Fecha de los datos
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 
    
    # Fecha del último dato
    t = pd.to_datetime(Dates[-1])
    
    # Valor del último dato
    Dato = Values[-1]
    
    # Se guardan la fecha y el valor del dato  
    data = dict(x=[t], y=[Dato])

    return  serialize_json(data)
     
	#################### Gráfico de Nivel de cama  ################################ 
	##################################################################################
	
@app.route('/make_ajax_plot_1_7', methods=['POST'])
def make_ajax_plot_1_7(Escala,resample):
	
# Esta función genera un gráfico de la variable "Nivel de cama"	
	
	# Se ajustan las funcionalidades que tendrá el gráfico
    TOOLTIPS = [
            ("fecha", "@x{%F}"),
            ("hora", "@x{%R}"),
            ("valor", "@y{0.0}"),
        ]      
	
    TOOLS="pan,wheel_zoom,box_zoom,reset,save"                
    
    # Se crea la variable que contiene la información del gráfico    
    plot = figure(plot_height=450, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)

    ###########################################################
    ## Obtención de los datos de predicciones pasadas
    
    source1 = AjaxDataSource(data_url=request.url_root + 'get_source_past_forecast_controller_f_bed_optimal/', polling_interval=int(10000), mode='append' )
    
    # Se obtienen los datos de las predicciones pasadas    
    df2 = get_source_past_forecast_controller(Escala , "f_bed_optimal",resample)
	
	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar    
    Values = df2['value'].values[:] 

    # Se guardan las fechas y valores    
    source1.data = dict(x=Dates, y=Values) 
    
	# Se asigna una curva y puntos a los valores obtenidos    
    r3 = plot.line('x', 'y', source=source1, line_width=2, color = 'green')
    r4 = plot.circle('x','y', size=5, source=source1, color="green", fill_alpha=1)
    
    ###########################################################
    ## Obtención de las predicciones que se generan de la variable

    # Variable que guarda los datos de la predicción y se actualiza completamente cada 10 segundos    
    source3 = AjaxDataSource(data_url=request.url_root + 'data_forecast_controller_f_bed_optimal/', polling_interval=int(10000), mode='replace' , max_size = 18 )

	# Se obtienen los datos de las predicciones	
    df2 = get_source_forecast_controller(Escala , "f_bed_optimal")

    # Fecha de la predicción    
    Dates = df2['datetime'].values[:]

    # Valor de cada dato a graficar    
    Values = df2['value'].values[:] 
    
    # Variable que se usará para asignar una fecha y hora para cada punto a graficar
    t = []

    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos    
    for i in range(len( Values[0] )):
     t.append(		pd.to_datetime(Dates[0] + np.timedelta64(i*5,'m'))		)

    # Se guardan las fechas y valores    
    source3.data = dict(x=t, y=Values[0]) 
    
    # Se asigna una curva y puntos a los valores obtenidos    
    r1 = plot.line('x', 'y', source=source3, line_width=2, color = 'green')
    r2 = plot.circle('x','y', size=5, source=source3, color="green", fill_alpha=1)
    
    ###########################################################
    ## Obtención de los valores pasados obtenidos de los sensores
    
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)    
    source4 = AjaxDataSource(data_url=request.url_root + 'data_past_sensor_bo_7110_lt_1009_s4/', polling_interval=int(10000), mode='append' )

	# Se obtienen los datos pasados de los sensores
    df2 = get_source_sensor(Escala , "bo_7110_lt_1009_s4",resample,0)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]

    # Valor de cada dato a graficar    
    Values = df2['value'].values[:] 

    # Se guardan las fechas y valores    
    source4.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r5 = plot.line(x='x', y='y', source=source4, line_width=2, color = 'red')
    r6 = plot.circle(x='x', y='y', size=5, source=source4, color="red", fill_alpha=1)
	###########################################################
    ## Se ajustan los parámetros que ajustan la información de los ejes del gráfico

    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter(seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"x": 'datetime'} ))     
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"

    # Se define la leyenda del gráfico
    legend = Legend(items=[
    ("Nivel de cama predicho"   , [r1, r2,r3,r4]),
    ("Nivel de cama real" , [r5,r6]),
     ],label_text_font= "bookman", location=(0, 0),orientation = "horizontal",click_policy = "hide")
    plot.add_layout(legend, 'below')   

    script, div = components(plot)
    return script, div

@app.route('/get_source_past_forecast_controller_f_bed_optimal/', methods=['POST'])
def get_source_past_forecast_controller_f_bed_optimal():
# Esta función obtiene el último valor pasado de la predicción de la variable "Nivel de cama"
 
	# Se obtienen los datos pasados de las predicciones
    df2 = get_source_past_forecast_controller(2 , "f_bed_optimal",5/60)

    # Fechas de los datos    
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos
    Values = df2['value'].values[:] 

    # Fecha del último dato    
    t = pd.to_datetime(Dates[-1])
    
    # Valor del último dato
    Dato = Values[-1]
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_forecast_controller_f_bed_optimal/', methods=['POST'])
def data_forecast_controller_f_bed_optimal():

	# Esta función obtiene la última predicción de la variable "Nivel de cama"
	
	# Se obtienen los datos de la predicción	
    df2 = get_source_forecast_controller(2 , "f_bed_optimal")

    # Fechas de los datos    
    Dates = df2['datetime'].values[:]
    
    # Valor de los datos    
    Values = df2['value'].values[:] 

    # Variable que se usará para asignar una fecha y hora para cada punto a graficar    
    t = []

    # Se generan 18 datos de fechas que van desde la fecha de la predicción y se espacian temporalmente 5 minutos    
    for i in range(len( Values[0] )):
     t.append(	datetime.datetime.now(tz = pytz.timezone('America/Caracas')) + datetime.timedelta(minutes=i*5))

    # Valores de la última predicción    
    Dato = Values[0]
    
    # Se guardan la fecha y el valor del dato
    data = dict(x=t, y=Dato)
    
    return  serialize_json(data)    

@app.route('/data_past_sensor_bo_7110_lt_1009_s4/', methods=['POST'])
def data_past_sensor_bo_7110_lt_1009_s4():

# Esta función obtiene el último valor pasado de la variable "Nivel de cama"
 
	# Se obtienen los datos de los valores pasado
    df2 = get_source_sensor(2 , "bo_7110_lt_1009_s4",0,0)

    # Fecha de los datos    
    Dates = df2['datetime'].values[:]

    # Valor de los datos
    Values = df2['value'].values[:] 

    # Fecha del último dato    
    t = pd.to_datetime(Dates[-1])

    # Valor del último dato
    Dato = Values[-1]

    # Se guardan la fecha y el valor del dato      
    data = dict(x=[t], y=[Dato])

    return  serialize_json(data)
   
	#################### Gráfico de Setpoint flujo floculante ################################ 
	##################################################################################
@app.route('/make_ajax_plot_1_8', methods=['POST'])
def make_ajax_plot_1_8(Escala,resample):

# Esta función genera un gráfico de la variable "Setpoint flujo floculante"
       
    # Se ajustan las funcionalidades que tendrá el gráfico
    TOOLTIPS = [
            ("fecha", "@x{%F}"),
            ("hora", "@x{%R}"),
            ("valor", "@y{0.0}"),
        ]      	
	
    TOOLS="pan,wheel_zoom,box_zoom,reset,save"                

    # Se crea la variable que contiene la información del gráfico  
    plot = figure(plot_height=450, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)

    ###########################################################
    ## Obtención de los datos del setpoint del controlador
    
    # Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)        
    source1 = AjaxDataSource(data_url=request.url_root + 'get_source_sensor_controller_cb_7120_fic_1002_sp_ext/', polling_interval=int(10000), mode='append' )

    df2 = get_source_sensor(Escala , "cb_7120_fic_1002_sp_ext",resample,0)
	
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    source1.data = dict(x=Dates, y=Values) 
    
    r1 = plot.line('x', 'y', source=source1, line_width=2, color = 'green')
    r2 = plot.circle('x','y', size=5, source=source1, color="green", fill_alpha=1)
    
    ###########################################################
    ## Obtención de los datos del setpoint real (lo que realmente se midió)
        
	# Variable que guarda los datos pasados y agrega un nuevo dato cada 10 segundos (sin borrar datos anteriores)   
    source4 = AjaxDataSource(data_url=request.url_root + 'data_past_sensor_br_7120_ft_1002/', polling_interval=int(10000), mode='append' )

	 # Se obtienen los datos pasados de los sensores
    df2 = get_source_sensor(Escala , "br_7120_ft_1002",resample,0)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]

    # Valor de cada dato a graficar    
    Values = df2['value'].values[:] 

    # Se guardan las fechas y valores    
    source4.data = dict(x=Dates, y=Values) 

    # Se asigna una curva y puntos a los valores obtenidos    
    r3 = plot.line('x', 'y', source=source4, line_width=2, color = 'red')
    r4 = plot.circle('x', 'y', size=5, source=source4, color="red", fill_alpha=1)
	###########################################################	
    ## Se ajustan los parámetros que ajustan la información de los ejes del gráfico

    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter(seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"x": 'datetime'} ))     
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"

    # Se define la leyenda del gráfico
    legend = Legend(items=[
    ("Setpoint flujo floculante"   , [r1, r2]),
    ("Flujo Floculante real" , [r3,r4]),
     ],label_text_font= "bookman", location=(0, 0),orientation = "horizontal",click_policy = "hide")
    plot.add_layout(legend, 'below')   

    script, div = components(plot)
    return script, div

@app.route('/get_source_sensor_controller_cb_7120_fic_1002_sp_ext/', methods=['POST'])
def get_source_sensor_controller_cb_7120_fic_1002_sp_ext():
 
 # Esta función obtiene el último valor pasado de la variable "Setpoint flujo floculante"

	# Se obtienen los datos de los valores pasado
    df2 = get_source_sensor(2 , "cb_7120_fic_1002_sp_ext",0,0)

    # Fecha de los datos    
    Dates = df2['datetime'].values[:]

    # Valor de los datos    
    Values = df2['value'].values[:] 

    # Fecha del último dato    
    t = pd.to_datetime(Dates[-1]) 
    
    # Valor del último dato    
    Dato = Values[-1] 

    # Se guardan la fecha y el valor del dato      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

####################################################################################
###############  Ajax querys que no han sido utilizadas ############################
####################################################################################
   
@app.route('/data_past_controller_bk_7110_ft_1030/', methods=['POST'])
def data_past_controller_bk_7110_ft_1030():
 
    df2 = get_source_past_controller(2 , "bk_7110_ft_1030")
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])

    return  serialize_json(data)
  
@app.route('/data_past_controller_cb_7120_fic_1002_sp_ext/', methods=['POST'])
def data_past_controller_cb_7120_fic_1002_sp_ext():
 
    df2 = get_source_past_controller(2 , "cb_7120_fic_1002_sp_ext")
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_controller_cc_4310_pu_0003_sp_ext/', methods=['POST'])
def data_past_controller_cc_4310_pu_0003_sp_ext():
 

    df2 = get_source_past_controller(2 , "cc_4310_pu_0003_sp_ext")
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_controller_cd_4310_pu_0004_sp_ext/', methods=['POST'])
def data_past_controller_cd_4310_pu_0004_sp_ext():
 

    df2 = get_source_past_controller(2 , "cd_4310_pu_0004_sp_ext")
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)
    
@app.route('/data_past_sensor_cc_4310_pu_0003_sp_ext/', methods=['POST'])
def data_past_sensor_cc_4310_pu_0003_sp_ext():
 
    df2 = get_source_sensor(2 , "cc_4310_pu_0003_sp_ext",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_cd_4310_pu_0004_sp_ext/', methods=['POST'])
def data_past_sensor_cd_4310_pu_0004_sp_ext():
 
    df2 = get_source_sensor(2 , "cd_4310_pu_0004_sp_ext",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/get_source_past_forecast_controller_cc_4310_pu_0003_sp_ext/', methods=['POST'])
def get_source_past_forecast_controller_cc_4310_pu_0003_sp_ext():
 
    df2 = get_source_past_forecast_controller(2 , "cc_4310_pu_0003_sp_ext",0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)    

@app.route('/get_source_past_forecast_controller_cd_4310_pu_0004_sp_ext/', methods=['POST'])
def get_source_past_forecast_controller_cd_4310_pu_0004_sp_ext():
 
    df2 = get_source_past_forecast_controller(2 , "cd_4310_pu_0004_sp_ext",0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)    

####################################################################################
###############  Ajax querys que se utilizan opr los gráficos de la pestaña "Espesador" ############################
####################################################################################

@app.route('/data_past_sensor_bf_7110_dt_1011/', methods=['POST'])
def data_past_sensor_bf_7110_dt_1011():
 
    df2 = get_source_sensor(2 , "bf_7110_dt_1011",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bg_7110_dt_1011_solido/', methods=['POST'])
def data_past_sensor_bg_7110_dt_1011_solido():
 
    df2 = get_source_sensor(2 , "bg_7110_dt_1011_solido",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)
    
@app.route('/data_past_sensor_bh_7110_dt_1030/', methods=['POST'])
def data_past_sensor_bh_7110_dt_1030():
 
    df2 = get_source_sensor(2 , "bh_7110_dt_1030",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bj_7110_ft_1012/', methods=['POST'])
def data_past_sensor_bj_7110_ft_1012():
 
    df2 = get_source_sensor(2 , "bj_7110_ft_1012",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bl_7110_lt_1009/', methods=['POST'])
def data_past_sensor_bl_7110_lt_1009():
 
    df2 = get_source_sensor(2 , "bl_7110_lt_1009",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)
    
@app.route('/data_past_sensor_bm_7110_lt_1009_s2/', methods=['POST'])
def data_past_sensor_bm_7110_lt_1009_s2():
 
    df2 = get_source_sensor(2 , "bm_7110_lt_1009_s2",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bn_7110_lt_1009_s3/', methods=['POST'])
def data_past_sensor_bn_7110_lt_1009_s3():
 
    df2 = get_source_sensor(2 , "bn_7110_lt_1009_s3",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bp_7110_ot_1003/', methods=['POST'])
def data_past_sensor_bp_7110_ot_1003():
 
    df2 = get_source_sensor(2 , "bp_7110_ot_1003",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bt_7120_lit_001/', methods=['POST'])
def data_past_sensor_bt_7120_lit_001():
 
    df2 = get_source_sensor(2 , "cd_4310_pu_0004_sp_ext",0,)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bs_7120_ft_1001/', methods=['POST'])
def data_past_sensor_bs_7120_ft_1001():
 
    df2 = get_source_sensor(2 , "bs_7120_ft_1001",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_ci_4310_pt_0023/', methods=['POST'])
def data_past_sensor_ci_4310_pt_0023():
 
    df2 = get_source_sensor(2 , "ci_4310_pt_0023",0,0)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

####################################################################################
###############  Ajax querys que se utilizan opr los gráficos de la pestaña "Espesador Prep." ############################
####################################################################################

@app.route('/data_past_sensor_bq_7110_pt_1010_prepro/', methods=['POST'])
def data_past_sensor_bq_7110_pt_1010_prepro():
 
    df2 = get_source_sensor(2 , "bq_7110_pt_1010",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bi_7110_dt_1030_solido_prepro/', methods=['POST'])
def data_past_sensor_bi_7110_dt_1030_solido_prepro():
 
    df2 = get_source_sensor(2 , "bi_7110_dt_1030_solido",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bo_7110_lt_1009_s4_prepro/', methods=['POST'])
def data_past_sensor_bo_7110_lt_1009_s4_prepro():
 
    df2 = get_source_sensor(2 , "bo_7110_lt_1009_s4",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_br_7120_ft_1002_prepro/', methods=['POST'])
def data_past_sensor_br_7120_ft_1002_prepro():
 
    df2 = get_source_sensor(2 , "br_7120_ft_1002",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bk_7110_ft_1030_prepro/', methods=['POST'])
def data_past_sensor_bk_7110_ft_1030_prepro():
 
    df2 = get_source_sensor(2 , "bk_7110_ft_1030",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bf_7110_dt_1011_prepro/', methods=['POST'])
def data_past_sensor_bf_7110_dt_1011_prepro():
 
    df2 = get_source_sensor(2 , "bf_7110_dt_1011",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bg_7110_dt_1011_solido_prepro/', methods=['POST'])
def data_past_sensor_bg_7110_dt_1011_solido_prepro():
 
    df2 = get_source_sensor(2 , "bg_7110_dt_1011_solido",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)
    
@app.route('/data_past_sensor_bh_7110_dt_1030_prepro/', methods=['POST'])
def data_past_sensor_bh_7110_dt_1030_prepro():
 
    df2 = get_source_sensor(2 , "bh_7110_dt_1030",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bj_7110_ft_1012_prepro/', methods=['POST'])
def data_past_sensor_bj_7110_ft_1012_prepro():
 
    df2 = get_source_sensor(2 , "bj_7110_ft_1012",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bl_7110_lt_1009_prepro/', methods=['POST'])
def data_past_sensor_bl_7110_lt_1009_prepro():
 
    df2 = get_source_sensor(2 , "bl_7110_lt_1009",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)
    
@app.route('/data_past_sensor_bm_7110_lt_1009_s2_prepro/', methods=['POST'])
def data_past_sensor_bm_7110_lt_1009_s2_prepro():
 
    df2 = get_source_sensor(2 , "bm_7110_lt_1009_s2",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bn_7110_lt_1009_s3_prepro/', methods=['POST'])
def data_past_sensor_bn_7110_lt_1009_s3_prepro():
 
    df2 = get_source_sensor(2 , "bn_7110_lt_1009_s3",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bp_7110_ot_1003_prepro/', methods=['POST'])
def data_past_sensor_bp_7110_ot_1003_prepro():
 
    df2 = get_source_sensor(2 , "bp_7110_ot_1003",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bt_7120_lit_001_prepro/', methods=['POST'])
def data_past_sensor_bt_7120_lit_001_prepro():
 
    df2 = get_source_sensor(2 , "cd_4310_pu_0004_sp_ext",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_bs_7120_ft_1001_prepro/', methods=['POST'])
def data_past_sensor_bs_7120_ft_1001_prepro():
 
    df2 = get_source_sensor(2 , "bs_7120_ft_1001",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)

@app.route('/data_past_sensor_ci_4310_pt_0023_prepro/', methods=['POST'])
def data_past_sensor_ci_4310_pt_0023_prepro():
 
    df2 = get_source_sensor(2 , "ci_4310_pt_0023",0,1)
    
    Dates = df2['datetime'].values[:]
    Values = df2['value'].values[:] 
    
    t = pd.to_datetime(Dates[-1]) #
    Dato = Values[-1] #
      
    data = dict(x=[t], y=[Dato])
    
    return  serialize_json(data)


####################################################################################
####################################################################################
####################################################################################
	    
@app.route('/_stream', methods=['POST'])
def stream():

# Esta función entrega los gráficos de la pestaña "Estación Meteorológica"
	
    try:
		# Primero se intenta generar el gráfico completo (datos pasados y predicciones)
        if request.method == 'POST':
            seleccion = request.form['seleccion']

            if seleccion == "all":
				# Creación de los tres gráficos
                p1 = create_fig(49, 73, "6_5", "Tiempo", "Velocidad viento [m/s]", "orange","tomato")
                p2 = create_fig(49, 73, "18_506", "Tiempo", "Temperatura [°C]", "lightseagreen","teal")
                p3 = create_fig(49, 73, "19_507", "Tiempo", "Humedad realtiva [%]", "green","lime")
                
                script1, div1 = components(p1)
                script2, div2 = components(p2)
                script3, div3 = components(p3)

            return jsonify(script1=script1, div1=div1,script2=script2, div2=div2,script3=script3, div3=div3)
 
    except Exception as e:
     pass
        
    try:
		# Se intenta generar el gráfico solo con datos de las predicciones
        if request.method == 'POST':
            seleccion = request.form['seleccion']
            if seleccion == "all":
				# Creación de los tres gráficos
                p1 = create_fig(0, 73, "6_5", "Tiempo", "Velocidad viento [m/s]", "orange","tomato")
                p2 = create_fig(0, 73, "18_506", "Tiempo", "Temperatura [°C]", "lightseagreen","teal")
                p3 = create_fig(0, 73, "19_507", "Tiempo", "Humedad realtiva [%]", "green","lime")

                script1, div1 = components(p1)
                script2, div2 = components(p2)
                script3, div3 = components(p3)
            return jsonify(script1=script1, div1=div1,script2=script2, div2=div2,script3=script3, div3=div3)

    except Exception as e:
     pass 
     
    try:
		# Se intenta cargar el gráfico solo con datos pasados
        if request.method == 'POST':
            seleccion = request.form['seleccion']
            if seleccion == "all":
				# Creación de los tres gráficos
                p1 = create_fig(49, 0, "6_5", "Tiempo", "Velocidad viento [m/s]", "orange","tomato")
                p2 = create_fig(49, 0, "18_506", "Tiempo", "Temperatura [°C]", "lightseagreen","teal")
                p3 = create_fig(49, 0, "19_507", "Tiempo", "Humedad realtiva [%]", "green","lime")

                script1, div1 = components(p1)
                script2, div2 = components(p2)
                script3, div3 = components(p3)
            return jsonify(script1=script1, div1=div1,script2=script2, div2=div2,script3=script3, div3=div3)

    except Exception as e:
     pass 
        
    try:
		# Luego de que fallan todos los intentos anteriores se procede a generar gráficos vacíos
        if request.method == 'POST':
            seleccion = request.form['seleccion']

            if seleccion == "all":
				# Creación de los tres gráficos
                p1 = create_fig_without_line( "Tiempo", "Velocidad viento [m/s]")
                p2 = create_fig_without_line( "Tiempo", "Temperatura [°C]")
                p3 = create_fig_without_line( "Tiempo", "Humedad realtiva [%]")

                script1, div1 = components(p1)
                script2, div2 = components(p2)
                script3, div3 = components(p3)
            return jsonify(script1=script1, div1=div1,script2=script2, div2=div2,script3=script3, div3=div3)
    except Exception as e:
     return str(e)

@app.route('/_forecast_alarm', methods=['POST'])
def forecast_alarm():
    try:
		# Esta función entrega las predicciones de la probabilidad de lluvia y la temperatura, junto con las fechas de los datos
				
        temp = get_source_forecast(24 * 5, 'Temperature')
        prec = get_source_forecast(24 * 5, 'Probability of Prec.')
        
        # Si no hay datos de temperatura o de precipitaciones se definen ambos datos con valor cero y la fcha actual
        if (temp.empty) or (prec.empty):
			
         date_temp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+0000')
         date_prec = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+0000')
         return jsonify(date_temp=date_temp, prob_temp=0,
                       date_prec=date_prec, prob_prec=0
                       ) 
                       
        else:
		 
		 # indice del max de los valores
         i = temp["value"].idxmax()  
         
         # se obtiene fecha
         date_temp = temp.iat[i, 0]  
         
         # se obtiene valor
         value_temp = temp.iat[i, 1]  
         
         # indice del max de los valores
         j = prec["value"].idxmax()  
         
         date_prec = prec.iat[j, 0]
         value_prec = prec.iat[j, 1]

         return jsonify(date_temp=date_temp.strftime('%d %b %Y %H:%M'), prob_temp=value_temp,
						   date_prec=date_prec.strftime('%d %b %Y %H:%M'), prob_prec=value_prec
						   )
    except Exception as e:
     return str(e)

def create_fig(past_hours, forecast_hours, sensor_code, x_axis_label, y_axis_label,past_color, forecast_color):
    try:
		# Esta función genera un gráfico con los datos pasados y futuros del sensor con nombre "sensor_code"
		# Esta función se utiliza para generar los gráficos de la pestaña "Estación Meteorológica"
		
		# Variables que modifican la visualización del gráfico
        hover_lin_width = 5
        circle_size = 4
        circle_alpha = 1
        hover_line_alph = 0.7
        
        # Se ajustan las opciones que tendrá el gráfico
        TOOLS="pan,wheel_zoom,box_zoom,reset,save"

        TOOLTIPS = [
            ("fecha", "@datetime{%F}"),
            ("hora", "@datetime{%R}"),
            ("valor", "@value{0.0}"),
        ]

		# Variable que contendrá al gráfico
        plot = figure(plot_width=380, plot_height=260, x_axis_type="datetime", sizing_mode='scale_width',tools=TOOLS)
        
        # Se ajustan los ejes del gráfico
        plot.xaxis.axis_label_text_font = "bookman"
        plot.yaxis.axis_label_text_font = "bookman"
        plot.title.align = 'center'
        plot.xaxis.formatter = DatetimeTickFormatter(seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
        plot.xaxis.major_label_orientation = -np.pi/4

		# Se revisa si se requieren datos pasados (past_hours > 0)
        if (past_hours > 0):

		 # Se solicitan y guardan en una variable los datos pasados
         df1 = get_source(past_hours, sensor_code)
         date_df1 = df1.tail(1)

		 # Variable que guarda las fechas
         Dates = df1['datetime'].values[:]
         
         # Variable que guarda los valores
         Values = df1['value'].values[:]           
         
         # Se insertan puntos en el origen para asegurar que las líneas del gráfico representen una línea poligonal cerrada, para después poder cambiar el color del área encerrada
         Dates = np.insert(Dates  , 0 , [Dates[0]])
         Dates = np.insert(Dates  , np.size(Dates) , [Dates[np.size(Dates)-1]])
         Values = np.insert(Values , 0 , 0)
         Values = np.insert(Values , np.size(Values) , 0)

         source1 = ColumnDataSource(df1)
         
         # Se genera el gráfico correspondiente a los datos pasados (línea y círculos señalando donde hay un dato)
         r1 = plot.circle(x='datetime', y='value', size=circle_size, source=source1, color="green", fill_alpha=circle_alpha)
         r2 = plot.line(x='datetime', y='value', line_width=4, source=source1, color=past_color, line_alpha=0.5,
            hover_alpha=hover_line_alph, hover_line_color=past_color) 
         
         # Se colorea el área encerrada por el gráfico
         plot.patch(  Dates,Values  , color=past_color, alpha=0.3, line_width=2)	
         
         # Se activa la opción "hover"
         r2.hover_glyph.line_width=hover_lin_width
        
        # Se revisa si se requieren datos de predicciones (forecast_hours > 0)     
        if (forecast_hours > 0):
         
         # Se solicitan y guardan en una variable los datos de predicciones
         df2 = get_source_forecast(forecast_hours, sensor_code)

		 # Si se solicitaron datos pasados se concatenan los datos pasados y futuros
         if past_hours > 0:
           past_hours = past_hours + 24

		   # Se concatenan los datos pasados y futuros 
           df1_concat_df2 = pd.concat([date_df1, df2])
           df1_concat_df2 = df1_concat_df2.reset_index(drop=True)
           
           # Variable que guarda las fechas
           Dates = df1_concat_df2['datetime'].values[:]
           
           # Variable que guarda los valores
           Values = df1_concat_df2['value'].values[:]

           # Se insertan puntos en el origen para asegurar que las líneas del gráfico representen una línea poligonal cerrada, para después poder cambiar el color del área encerrada                      
           Dates = np.insert(Dates  , 0 , [Dates[0]])
           Dates = np.insert(Dates  , np.size(Dates) , [Dates[np.size(Dates)-1]])
           Values = np.insert(Values , 0 , 0)
           Values = np.insert(Values , np.size(Values) , 0)
           
           
           source2 = ColumnDataSource(df1_concat_df2)
		   
		   # Se genera el gráfico correspondiente a los datos pasados (línea y círculos señalando donde hay un dato)	
           r3 = plot.circle(x='datetime', y='value', size=circle_size, source=source2, color="green", fill_alpha=circle_alpha)
           r4 = plot.line(x='datetime', y='value', line_width=3, source=source2, color=forecast_color, line_alpha=0.6,
            hover_alpha=hover_line_alph, hover_line_color=forecast_color)

           # Se colorea el área encerrada por el gráfico
           plot.patch(  Dates,Values  , color=forecast_color, alpha=0.3, line_width=2)	
           
         # Si no se solicitaron datos pasados se utilizan los datos de predicciones y no se concatenan
         else:
				
           source2 = ColumnDataSource(df2)
           
           # Variable que guarda las fechas
           Dates = df2['datetime'].values[:]
           
           # Variable que guarda los valores
           Values = df2['value'].values[:]
           
           # Se insertan puntos en el origen para asegurar que las líneas del gráfico representen una línea poligonal cerrada, para después poder cambiar el color del área encerrada           
           Dates = np.insert(Dates  , 0 , [Dates[0]])
           Dates = np.insert(Dates  , np.size(Dates) , [Dates[np.size(Dates)-1]])
           Values = np.insert(Values , 0 , 0)
           Values = np.insert(Values , np.size(Values) , 0)

           # Se genera el gráfico correspondiente a los datos pasados (línea y círculos señalando donde hay un dato)
           r3 = plot.circle(x='datetime', y='value', size=circle_size, source=source2, color="green", fill_alpha=circle_alpha, legend='Pred')
           r4 = plot.line(x='datetime', y='value', line_width=4, source=source2, color=forecast_color, line_alpha=0.6, legend='Pred',
            hover_alpha=hover_line_alph, hover_line_color=forecast_color)
           
           # Se colorea el área encerrada por el gráfico 
           plot.patch(  Dates,Values  , color=forecast_color, alpha=0.3, line_width=2)	
         
         # Se activa la opción "hover"
         r4.hover_glyph.line_width=hover_lin_width
         
        plot.toolbar.logo = None  # elimina logo bokeh
        
        # Se selecciona el texto que irá en el gráfico como leyenda según los datos solicitados
        if (past_hours > 0) and (forecast_hours > 0):
			
         legend = Legend(items=[
         ("Real"   , [r1, r2]),
         ("Predicho" , [r3,r4]),
         ], label_text_font= "bookman",location=(0, 0),orientation = "horizontal",click_policy = "hide")

        elif (past_hours > 0) and (forecast_hours <= 0):

         legend = Legend(items=[
         ("Real"   , [r1, r2]),
         ], label_text_font= "bookman",location=(0, 0),orientation = "horizontal",click_policy = "hide")
			
        elif (past_hours <= 0) and (forecast_hours > 0):

         legend = Legend(items=[
         ("Predicho" , [r3,r4]),
         ], label_text_font= "bookman",location=(0, 0),orientation = "horizontal",click_policy = "hide")

		# Se añade la leyenda al gráfico
        plot.add_layout(legend, 'below')	           
    
        plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters={"datetime": 'datetime'}))
        
        return plot

    except Exception as e:
     return str(e)

def create_fig_without_line( x_axis_label, y_axis_label):
    try:
		# Esta función crea un gráfico vacío, se utiliza a esta función en la pestaña "Estación Meteorológica"
		
        TOOLTIPS = [
            ("fecha", "@datetime{%F}"),
            ("hora", "@datetime{%R}"),
            ("valor", "@value{0.0}"),
        ]

        plot = figure(plot_width=380, plot_height=260, x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                      x_axis_type="datetime", sizing_mode='scale_width')
        plot.xaxis[0].formatter = DatetimeTickFormatter(days='%b %d')
        
        # formatea mouse hover
        plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters={"datetime": 'datetime'}))

        plot.toolbar.logo = None  # elimina logo bokeh
        plot.toolbar_location = None  # elimina barra de herrmaientas bokeh

        return plot
    
    except Exception as e:
        return str(e)

def get_source_forecast_controller(hr, sensor_name):
    try:
		# Esta función obtiene las predicciones hechas por el controlador guardadas en tabla "fondef.pso_outputs"

		# Se obtiene la fecha actual
        utcNow = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+0000')
        
        # Se obtiene la fecha más antigua a buscar
        utcPast = datetime.datetime.utcnow() - datetime.timedelta(hours=hr)
        utcPast = utcPast.strftime('%Y-%m-%d %H:%M:%S+0000')
		
		# Se hace la petición a la tabla de cassandra
        rows2 = connection.execute(
            "select * from fondef.pso_outputs_saul_v2 where sensor='{}' and datetime <= '{}' and datetime >= '{}' ".format(
                sensor_name, utcNow, utcPast))		
        
        df = rows2._current_rows
        
        # Si no hay datos se entregan solo ceros como datos
        if df.empty:
         return Zero_filling_forecast(hr)
         
        else:
         
         dfrev = df.iloc[::1]

         dfrev['datetime'] = dfrev['datetime'].dt.tz_localize('utc').dt.tz_convert('America/Caracas')
         dfrev['datetime'] = dfrev['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+0000')
         dfrev['datetime'] = dfrev['datetime'].apply(lambda x:
														datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+0000'))
         dfrev = dfrev.reset_index(drop=True)
         frame = dfrev.loc[lambda df: dfrev['sensor'] == sensor_name, ['datetime', 'value']]

         return frame
         
    except Exception as e:
     return str(e)

def get_source_past_controller(hr, sensor_name):
    try:
		# Esta función genera datos pasados entregados por el controlador
		
        utcNow = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+0000')

        utcPast = datetime.datetime.utcnow() - datetime.timedelta(hours=hr)
        utcPast = utcPast.strftime('%Y-%m-%d %H:%M:%S+0000')
        
		# Se genera la hora local pasada suponiendo que la hora local no está necesariamente ajustada en "America/Caracas"
        localPast = datetime.datetime.now(tz = pytz.timezone('America/Caracas')) - datetime.timedelta(hours=hr)
        localPast = localPast.strftime('%Y-%m-%d %H:%M:%S+0000')
        localPast = datetime.datetime.strptime(localPast, '%Y-%m-%d %H:%M:%S+0000') # Variable necesaria para utilizar más adelante en Holes_filling() 

		# Se hace la petición a la tabla de cassandra
        rows2 = connection.execute(
            "select * from fondef.mpc_out_v2 where sensor='{}' and datetime <= '{}' and datetime >= '{}' ".format(
                sensor_name, utcNow, utcPast))
                
        df = rows2._current_rows

		# Si no hay datos se entregan solo ceros como datos
        if df.empty:
         return Zero_filling_past(datetime.datetime.now(),hr)
         
        else:
         
         dfrev = df.iloc[::-1]
         dfrev['datetime'] = dfrev['datetime'].dt.tz_localize('utc').dt.tz_convert('America/Caracas')
         dfrev['datetime'] = dfrev['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+0000')
         dfrev['datetime'] = dfrev['datetime'].apply(lambda x:
														datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+0000'))
         dfrev = dfrev.reset_index(drop=True)
         frame = dfrev.loc[lambda df: dfrev['sensor'] == sensor_name, ['datetime', 'value']]
         
         # Se revisa si es que no hay saltos temporales, si los hay se rellena con datos
         frame = Holes_filling(frame,hr,localPast, 5)
		
         return frame
         
    except Exception as e:
     return str(e)

def get_source_sensor(hr, sensor_name, resample, PreProc):
    try:
		# Esta función se encarga de obtener los datos pasados del sensor con tag igual a "sensor_name", la fecha solicitada es desde la actual hasta "hr" horas hacia atrás
		
		# Se obtiene la fecha actual
        utcNow = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+0000')
		
		# Se obtiene la fecha más antigua a buscar
        utcPast = datetime.datetime.utcnow() - datetime.timedelta(hours=hr)
        utcPast = utcPast.strftime('%Y-%m-%d %H:%M:%S+0000')
		
		# Se genera la hora local pasada suponiendo que la hora local no está necesariamente ajustada en "America/Caracas"
        localPast = datetime.datetime.now(tz = pytz.timezone('America/Caracas')) - datetime.timedelta(hours=hr)
        localPast = localPast.strftime('%Y-%m-%d %H:%M:%S+0000')
        localPast = datetime.datetime.strptime(localPast, '%Y-%m-%d %H:%M:%S+0000') # Variable necesaria para utilizar más adelante en Holes_filling() 

		# Se hace la petición a la tabla de cassandra
        if (PreProc == 1):
         rows2 = connection.execute(
            "select * from fondef.dcs_sensors_processed where sensor='{}' and datetime <= '{}' and datetime >= '{}' ".format(
                sensor_name, utcNow, utcPast))        
        else:
         rows2 = connection.execute(
            "select * from fondef.dcs_sensors where sensor='{}' and datetime <= '{}' and datetime >= '{}' ".format(
                sensor_name, utcNow, utcPast))        

        df = rows2._current_rows
                
        # Si no hay datos se entregan solo ceros como datos
        if df.empty:
         return Zero_filling_past(datetime.datetime.now(),hr)
        else:

         dfrev = df.iloc[::-1]
         
         dfrev['datetime'] = dfrev['datetime'].dt.tz_localize('utc').dt.tz_convert('America/Caracas')
         dfrev['datetime'] = dfrev['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+0000')
         dfrev['datetime'] = dfrev['datetime'].apply(lambda x:
														datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+0000'))
														
         dfrev = dfrev.reset_index(drop=True)
         frame = dfrev

		 # Se verifica si se solicita hacer un remuestreo de los datos
         if resample !=0:
		  
		  # Se revisa si es que no hay saltos temporales, si los hay se rellena con datos 
          frame = Holes_filling(frame,hr,localPast, 0.1)
          frame = frame.set_index('datetime', drop=True)
          
          # Se hace un remuestreo de los datos, se toma el primer valor de cada periodo de "resample" minutos ("resample" es un número)
          frame = frame.resample( str(resample) + 'T' ).first()      
          frame['datetime'] = frame.index
      
         return frame
         
    except Exception as e:
     return str(e)

def get_source(hr, sensor_code):
    try:
		# Esta función se encarga de obtener los datos pasados de los datos meteorológicos con tag igual a "sensor_name", la fecha solicitada es desde la actual hasta "hr" horas hacia atrás
	
        connection.row_factory = pandas_factory
        connection.default_fetch_size = None

		# Se obtiene la fecha actual
        utcNow = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+0000')

		# Se hace la petición a la tabla de cassandra
        rows1 = connection.execute(
            "select * from fondef.weather_station_values where sensor='{}' and statics='aver' and datetime <= '{}' LIMIT {}".format(
                sensor_code, utcNow, hr))
        
        
        df = rows1._current_rows
        dfrev = df.iloc[::-1]

        dfrev['datetime'] = dfrev['datetime'].dt.tz_localize('utc').dt.tz_convert('America/Caracas')
        dfrev['datetime'] = dfrev['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+0000')
        dfrev['datetime'] = dfrev['datetime'].apply(lambda x:
                                                    datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+0000'))
        dfrev = dfrev.reset_index(drop=True)
        frame = dfrev.loc[lambda df: dfrev['sensor'] == sensor_code, ['datetime', 'value']]

        return frame
        
    except Exception as e:
     return str(e)

def get_source_forecast(hr, sensor_code):
    try:
		# Esta función se encarga de obtener los datos de las predicciones de los datos meteorológicos
		
		# Se obtiene la fecha actual
        utcNow = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+0000')

		# Se obtiene la fecha a futuro hasta la que buscar
        utcFuture = datetime.datetime.utcnow() + datetime.timedelta(hours=hr)
        utcFuture = utcFuture.strftime('%Y-%m-%d %H:%M:%S+0000')

		# Se determina el nombre del sensor según el código recibido
        if sensor_code == "6_5":
            sensor_code = "Wind Speed"
        elif sensor_code == "18_506":
            sensor_code = "Temperature"
        elif sensor_code == "19_507":
            sensor_code = "Relative Humidity"
        elif sensor_code == "5_6":
            sensor_code = "Precipitation"

		# Se hace la petición a la tabla de cassandra
        rows2 = connection.execute(
            "select * from fondef.weather_forecast_values where sensor='{}' and datetime >= '{}' and datetime <= '{}' ".format(
                sensor_code, utcNow, utcFuture))

        df = rows2._current_rows
        
        # Si no hay datos se entregan el dataframe vacío
        if df.empty:
         return df
         
        else:

         dfrev = df.iloc[::-1]
         dfrev['datetime'] = dfrev['datetime'].dt.tz_localize('utc').dt.tz_convert('America/Caracas')
         dfrev['datetime'] = dfrev['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+0000')
         dfrev['datetime'] = dfrev['datetime'].apply(lambda x:
														datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+0000'))
         dfrev = dfrev.reset_index(drop=True)

         frame = dfrev.loc[lambda df: dfrev['sensor'] == sensor_code, ['datetime', 'value']]
         return frame
         
    except Exception as e:
     return str(e)

def get_source_past_forecast_controller(hr, sensor_name,resample):
    try:
		# Esta función se encarga de obtener los datos de las predicciones pasados hechas por el controlador

		# Se obtiene la fecha actual
        utcNow = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+0000')

		# Se obtiene la fecha más antigua a buscar
        utcPast = datetime.datetime.utcnow() - datetime.timedelta(hours=hr)
        utcPast = utcPast.strftime('%Y-%m-%d %H:%M:%S+0000')

		# Se genera la hora local pasada suponiendo que la hora local no está necesariamente ajustada en "America/Caracas"
        localPast = datetime.datetime.now(tz = pytz.timezone('America/Caracas')) - datetime.timedelta(hours=hr)
        localPast = localPast.strftime('%Y-%m-%d %H:%M:%S+0000')
        localPast = datetime.datetime.strptime(localPast, '%Y-%m-%d %H:%M:%S+0000') # Variable necesaria para utilizar más adelante en Holes_filling() 

		# Se hace la petición a la tabla de cassandra
        rows2 = connection.execute(
            "select * from fondef.pso_outputs_saul_v2 where sensor='{}' and datetime <= '{}' and datetime >= '{}' ".format(
                sensor_name, utcNow, utcPast))	

        df = rows2._current_rows

		# Si no hay datos se entregan solo ceros como datos
        if df.empty:
         return Zero_filling_past(datetime.datetime.now(),hr)
         
        else:
        
         dfrev = df.iloc[::-1]
         data = { 'datetime': dfrev.iloc[:,1], 'value':  dfrev.iloc[:,2].str[0]}
         dfrev = pd.DataFrame(data=data)

         dfrev['datetime'] = dfrev['datetime'].dt.tz_localize('utc').dt.tz_convert('America/Caracas')
         dfrev['datetime'] = dfrev['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+0000')
         dfrev['datetime'] = dfrev['datetime'].apply(lambda x:
														datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+0000'))
         dfrev = dfrev.reset_index(drop=True)
         frame = dfrev

		 # Se verifica si se solicita hacer un remuestreo de los datos
         if resample !=0:
			 
		  # Se revisa si es que no hay saltos temporales, si los hay se rellena con datos 
          frame = Holes_filling(frame,hr,localPast, resample)
          frame = frame.set_index('datetime', drop=True)
          
          # Se hace un remuestreo de los datos, se toma el primer valor de cada periodo de "resample" minutos ("resample" es un número)
          frame = frame.resample( str(resample) + 'T' ).first()      
          frame['datetime'] = frame.index
      
         return frame
         
    except Exception as e:
     return str(e)

def Zero_filling_past(Final_time,hr):
# Esta función crea un dataframe lleno de ceros desde "Final_time - hr" hasta "Final_time"

 hr = int(hr)
 Datetime_array = [Final_time - datetime.timedelta(hours=x) for x in range(0, hr)]
 data = { 'datetime': Datetime_array, 'value':  [0]*len(Datetime_array)}
 frame = pd.DataFrame(data=data)	
 
 return frame	

def Zero_filling_forecast(hr):
# Esta función crea un dataframe lleno de ceros desde la hora actual hasta "hr" horas hacias adelente
	
 Datetime_array = [datetime.datetime.now() - datetime.timedelta(hours=x) for x in range(0, hr)]
 data = { 'datetime': Datetime_array, 'value':  [[0 for col in range(18)] for row in range(len(Datetime_array))]}
 frame = pd.DataFrame(data=data)	
 
 return frame		

def Holes_filling(df, hr, initial_datetime, resample):

# Esta función rellena los vacíos de información que pudiesen existir en un dataframe, se realiza este procedimiento extendiendo hacia adelante o atrás (en el tiempo) los datos que se tienen 

 # Número de periodos que tendrá en dataframe final
 periods = int(	hr*60/resample)
 freq = str(resample) + 'T'
 
 date_index = pd.date_range(initial_datetime, periods= periods, freq=freq )
 df2 = df.set_index('datetime', drop=True) 
 
 # Se genera un nuevo dataframe con los datos antiguos más los nuevos 
 df2 = df2.reindex(date_index, axis= 'index', tolerance= str(resample)+'min' ,method = 'nearest')
 df2.index = range(len(df2))
 
 df2['value'] = df2['value'].interpolate(method='pad',limit_direction='both',limit = 100000000)
 df2['value'] = df2['value'].interpolate(method='linear',limit_direction='both',limit = 100000000)
 
 df2['datetime'] = date_index

 return df2

@app.route("/Graficos_Esp", methods=['POST'])
def Graficos_Esp():
    try:
		# Esta función se encarga de generar los gráficos que se muestran en la pestaña "Espesador"
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')
            
        else:
		
		 # Si se indica un "Ini_date" igual a cero significa que se pide el gráfico en modo "online"
         if request.values['Ini_date'] == '0':
			 
          script, div = Make_Plot_Now_Esp(request.values['Tag_name'], int(float(request.values['PreProc'])))
		 
		 # Se inicia el proceso para generar el gráfico "offline"
         else:
			
          local = pytz.timezone('America/Caracas')

		  # Se unen las fechas iniciales y finales con sus respectivas horas
          string_date_ini = request.values['Ini_date'] + " " + request.values['Ini_time']
          string_date_fin = request.values['Fin_date'] + " " + request.values['Fin_time'] 
                    
          string_date_ini = datetime.datetime.strptime(string_date_ini, "%Y-%m-%d %H:%M") 
          string_date_fin = datetime.datetime.strptime(string_date_fin, "%Y-%m-%d %H:%M")          
          
          # Se transforma la fecha y hora local iniciales a la zona horaria UTC 
          local_dt = local.localize(string_date_ini, is_dst=None)
          string_date_ini = local_dt.astimezone(pytz.utc)
          string_date_ini = string_date_ini.strftime ("%Y-%m-%d %H:%M")
          string_date_ini = datetime.datetime.strptime(string_date_ini, "%Y-%m-%d %H:%M") 

		  # Se transforma la fecha y hora local finales a la zona horaria UTC 
          local_dt = local.localize(string_date_fin, is_dst=None)
          string_date_fin = local_dt.astimezone(pytz.utc)
          string_date_fin = string_date_fin.strftime ("%Y-%m-%d %H:%M")
          string_date_fin = datetime.datetime.strptime(string_date_fin, "%Y-%m-%d %H:%M")                 
          
          # Se calcula la diferencia temporal entre la fecha inicial y final
          string_date_diff = string_date_fin - string_date_ini
          
          # Se genera el gráfico requerido
          script, div = Make_Plot_Past_Esp(string_date_ini , string_date_fin , string_date_diff , request.values['Tag_name'], int(float(request.values['PreProc'])) )          
                  
         return jsonify(script = script, div = div)
         
    except Exception as e:
        print(e)
        return render_template('login.html')

def get_source_sensor_Plot_Esp(utc_1, utc_2,utc_diff, sensor_name, PreProc):
    try:
		# Esta función obtiene los datos pasados de los sensores, se utiliza para los gráficos de la pestaña "espesador"


		# Se calculan las horas de diferencia entre la fecha inicial y la fecha final
        
        hr = int(utc_diff.total_seconds()/60/60)

		# Se hace la petición a la tabla de cassandra
        if (PreProc == 1):
         rows2 = connection.execute(
            "select * from fondef.dcs_sensors_processed where sensor='{}' and datetime <= '{}' and datetime >= '{}' ".format(
                sensor_name, utc_2.strftime('%Y-%m-%d %H:%M:%S+0000'), utc_1.strftime('%Y-%m-%d %H:%M:%S+0000')))  
        else:
         rows2 = connection.execute(
            "select * from fondef.dcs_sensors where sensor='{}' and datetime <= '{}' and datetime >= '{}' ".format(
                sensor_name, utc_2.strftime('%Y-%m-%d %H:%M:%S+0000'), utc_1.strftime('%Y-%m-%d %H:%M:%S+0000')))  

		# Se generan las horas locales
        local_1 = utc_1 - datetime.timedelta(hours=3)

        local_1 = utc_1.strftime ("%Y-%m-%d %H:%M")
        local_1 = datetime.datetime.strptime(local_1, "%Y-%m-%d %H:%M")
        local_1 = local_1.replace(tzinfo=pytz.utc)
        local_1 = local_1.astimezone(pytz.timezone('America/Caracas'))
        local_1 = local_1.strftime ("%Y-%m-%d %H:%M")
        local_1 = datetime.datetime.strptime(local_1, "%Y-%m-%d %H:%M")
        
        local_2 = utc_2.strftime ("%Y-%m-%d %H:%M")
        local_2 = datetime.datetime.strptime(local_2, "%Y-%m-%d %H:%M")
        local_2 = local_2.replace(tzinfo=pytz.utc)
        local_2 = local_2.astimezone(pytz.timezone('America/Caracas'))
        local_2 = local_2.strftime ("%Y-%m-%d %H:%M")
        local_2 = datetime.datetime.strptime(local_2, "%Y-%m-%d %H:%M")
        
        df = rows2._current_rows

		# Si no hay datos se entregan solo ceros como datos
        if df.empty:
         return Zero_filling_past(local_2,hr)

        else:
		
		 # Se calcula el parámetro para realizar el remuestreo
         resample = int(utc_diff.total_seconds()/60/60/ 24 * 10)
         dfrev = df.iloc[::-1]
         
         dfrev['datetime'] = dfrev['datetime'].dt.tz_localize('utc').dt.tz_convert('America/Caracas')
         dfrev['datetime'] = dfrev['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S+0000')
         dfrev['datetime'] = dfrev['datetime'].apply(lambda x:
														datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+0000'))
														
         dfrev = dfrev.reset_index(drop=True)
         frame = dfrev
         
         # Se revisa si es que no hay saltos temporales, si los hay se rellena con datos
         frame = Holes_filling(frame,int(hr),local_1, 0.1)
         frame = frame.set_index('datetime', drop=True)

		 # Se verifica si se solicita hacer un remuestreo de los datos
         if resample != 0:
			 
		  # Se hace un remuestreo de los datos, se toma el primer valor de cada periodo de "resample" minutos
          frame = frame.resample( str(resample) + 'T' ).first()      
		
         return frame

    except Exception as e:
     return str(e)

def Make_Plot_Past_Esp(string_date_ini,string_date_fin,string_date_diff,sensor_name, PreProc):
	
# Esta función genera un gráfico de los datos pasados de la variable "sensor_name", el gráfico generado se muestra en la pestaña "espesador"
	
	# Se ajustan las funcionalidades que tendrá el gráfico	
    TOOLTIPS = [
            ("fecha", "@datetime{%d-%m-%Y}"),
            ("hora", "@datetime{%H:%M}"),
            ("valor", "@value{0.0}"),
        ]        
    TOOLS="pan,wheel_zoom,box_zoom,reset,save"  
                  
    # Se crea la variable que contiene la información del gráfico
    plot = figure(plot_height=450, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)
    
    ###########################################################
    ## Obtención de los valores pasados obtenidos del sensor
    
    # Se obtienen los datos pasados de los sensores
    df2 = get_source_sensor_Plot_Esp(string_date_ini, string_date_fin,string_date_diff, sensor_name, PreProc)
    data = ColumnDataSource(df2)

	# Se asigna una curva y puntos a los valores obtenidos
    r5 = plot.line(x='datetime', y='value', source=data, line_width=2, color = 'red')
    r6 = plot.circle(x='datetime', y='value', size=5, source=data, color="red", fill_alpha=1)
	###########################################################
    ## Se ajustan los parámetros que ajustan la información de los ejes del gráfico

    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter(hourmin = ["%H:%M %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"datetime": 'datetime'} ))     
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"    

	# Se define la leyenda del gráfico
    legend = Legend(items=[
    (leyendas_plot(sensor_name) , [r5,r6]),
     ], label_text_font= "bookman",location=(0, 0),orientation = "horizontal",click_policy = "hide")
    plot.add_layout(legend, 'below')    
    
    script, div = components(plot)
    return script, div
    
def Make_Plot_Now_Esp(sensor_name, PreProc):
	
# Esta función genera un gráfico que se actualiza periódicamente de los datos actuales de la variable "sensor_name", el gráfico generado se muestra en la pestaña "espesador"
    
    # Se ajustan las funcionalidades que tendrá el gráfico
    TOOLTIPS = [
            ("fecha", "@x{%F}"),
            ("hora", "@x{%R}"),
            ("valor", "@y{0.0}"),
        ]        
    TOOLS="pan,wheel_zoom,box_zoom,reset,save"      

	# Se crea la variable que contiene la información del gráfico              
    plot = figure(plot_height=450, sizing_mode='scale_width', x_axis_type="datetime", tools=TOOLS)
    
    ###########################################################
    Ruta = 'data_past_sensor_' + sensor_name 
    
    if (PreProc == 1):
     Ruta = Ruta +'_prepro/'
    else:
     Ruta = Ruta + '/'	

    # Variable que agrega un nuevo dato al gráfico cada 10 segundos (sin borrar datos anteriores)   
    source4 = AjaxDataSource(data_url=request.url_root + Ruta, polling_interval=int(10000), mode='append' )

	# Se obtienen el último datos del sensor
    df2 = get_source_sensor(0.0025 , sensor_name ,0, PreProc)

	# Fecha de cada dato a graficar
    Dates = df2['datetime'].values[:]
    
    # Valor de cada dato a graficar
    Values = df2['value'].values[:] 
    
    # Se guardan las fechas y valores
    source4.data = dict(x=Dates, y=Values) 
    
    # Se asigna una curva y puntos a los valores obtenidos
    r5 = plot.line('x', 'y', source=source4, line_width=2, color = 'red')
    r6 = plot.circle('x', 'y', size=5, source=source4, color="red", fill_alpha=1)
	###########################################################
    ## Se ajustan los parámetros que ajustan la información de los ejes del gráfico

    plot.toolbar.logo = None  # elimina logo bokeh
    plot.xaxis.formatter = DatetimeTickFormatter( seconds = ["%H:%M:%S %d/%m/%y "] ,minsec = ["%H:%M%S %d/%m/%y "], minutes = ["%H:%M %d/%m/%y "], hourmin = ["%H:%M %d/%m/%y "], hours = ["%H:%M %d/%m/%y "],days = ["%d/%m/%y"])    
    plot.xaxis.major_label_orientation = -np.pi/4
    plot.add_tools(HoverTool(tooltips=TOOLTIPS, formatters = {"x": 'datetime'} ))     
    plot.xaxis.axis_label_text_font = "bookman"
    plot.yaxis.axis_label_text_font = "bookman"    
	
	# Se define la leyenda del gráfico
    legend = Legend(items=[
    (leyendas_plot(sensor_name) , [r5,r6]),
     ], label_text_font= "bookman",location=(0, 0),orientation = "horizontal",click_policy = "hide")
    plot.add_layout(legend, 'below')    
    
    script, div = components(plot)
    return script, div

@app.route("/Graficos_t1", methods=['POST'])
def Graficos_t1():
    try:
		# Esta función genera los gráficos de la pestaña "tendencias" con escala de tiempo de 1 día 
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')
            
        else:
			
			# Se define la escala y periodo de remuestreo de los datos que irán en los gráficos
            Escala = 24
            resample = 1 * 60
            
            # Se generan los gráficos
            script1, div1 = make_ajax_plot_1_1(Escala,resample)
            script2, div2 = make_ajax_plot_1_2(Escala,resample)
            script3, div3 = make_ajax_plot_1_3(Escala,resample)
            script4, div4 = make_ajax_plot_1_4(Escala,resample)
            script5, div5 = make_ajax_plot_1_5(Escala,resample)
            script6, div6 = make_ajax_plot_1_6(Escala,resample)
            script7, div7 = make_ajax_plot_1_7(Escala,resample)
            script8, div8 = make_ajax_plot_1_8(Escala,resample)
                        

            return jsonify(script1 = script1, div1 = div1, script2 = script2, div2 = div2,
													 script3 = script3, div3 = div3, script4 = script4, div4 = div4,
													 script5 = script5, div5 = div5, script6 = script6, div6 = div6, 
													 script7 = script7, div7 = div7,script8 = script8, div8 = div8)
		
    except Exception as e:
        return render_template('login.html')

@app.route("/Graficos_t2", methods=['POST'])
def Graficos_t2():
    try:
		# Esta función genera los gráficos de la pestaña "tendencias" con escala de tiempo de 3 día
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')
            
        else:
			
			# Se define la escala y periodo de remuestreo de los datos que irán en los gráficos
            Escala = 72
            resample = 3 * 60

			# Se generan los gráficos
            script1, div1 = make_ajax_plot_1_1(Escala,resample)
            script2, div2 = make_ajax_plot_1_2(Escala,resample)
            script3, div3 = make_ajax_plot_1_3(Escala,resample)
            script4, div4 = make_ajax_plot_1_4(Escala,resample)
            script5, div5 = make_ajax_plot_1_5(Escala,resample)
            script6, div6 = make_ajax_plot_1_6(Escala,resample)
            script7, div7 = make_ajax_plot_1_7(Escala,resample)
            script8, div8 = make_ajax_plot_1_8(Escala,resample)

            return jsonify(script1 = script1, div1 = div1, script2 = script2, div2 = div2,
													 script3 = script3, div3 = div3, script4 = script4, div4 = div4,
													 script5 = script5, div5 = div5, script6 = script6, div6 = div6, 
													 script7 = script7, div7 = div7,script8 = script8, div8 = div8)
		
    except Exception as e:
        return render_template('login.html')

@app.route("/Graficos_t3", methods=['POST'])
def Graficos_t3():
    try:
		# Esta función genera los gráficos de la pestaña "tendencias" con escala de tiempo de 7 día
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')

        else:
			
			# Se define la escala y periodo de remuestreo de los datos que irán en los gráficos
            Escala = 168
            resample = 7 * 60
            
            # Se generan los gráficos
            script1, div1 = make_ajax_plot_1_1(Escala,resample)
            script2, div2 = make_ajax_plot_1_2(Escala,resample)
            script3, div3 = make_ajax_plot_1_3(Escala,resample)
            script4, div4 = make_ajax_plot_1_4(Escala,resample)
            script5, div5 = make_ajax_plot_1_5(Escala,resample)
            script6, div6 = make_ajax_plot_1_6(Escala,resample)
            script7, div7 = make_ajax_plot_1_7(Escala,resample)
            script8, div8 = make_ajax_plot_1_8(Escala,resample)
                        

            return jsonify(script1 = script1, div1 = div1, script2 = script2, div2 = div2,
													 script3 = script3, div3 = div3, script4 = script4, div4 = div4,
													 script5 = script5, div5 = div5, script6 = script6, div6 = div6, 
													 script7 = script7, div7 = div7,script8 = script8, div8 = div8)
		
    except Exception as e:
        return render_template('login.html')

@app.route("/Graficos_t4", methods=['POST'])
def Graficos_t4():
    try:
		# Esta función genera los gráficos de la pestaña "tendencias" con escala de tiempo de 1 hora
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')

        else:
			
			# Se define la escala y periodo de remuestreo de los datos que irán en los gráficos
            Escala = 1
            resample = 1
            
            # Se generan los gráficos
            script1, div1 = make_ajax_plot_1_1(Escala,resample)
            script2, div2 = make_ajax_plot_1_2(Escala,resample)
            script3, div3 = make_ajax_plot_1_3(Escala,resample)
            script4, div4 = make_ajax_plot_1_4(Escala,resample)
            script5, div5 = make_ajax_plot_1_5(Escala,resample)
            script6, div6 = make_ajax_plot_1_6(Escala,resample)
            script7, div7 = make_ajax_plot_1_7(Escala,resample)
            script8, div8 = make_ajax_plot_1_8(Escala,resample)
                        
            return jsonify(script1 = script1, div1 = div1, script2 = script2, div2 = div2,
													 script3 = script3, div3 = div3, script4 = script4, div4 = div4,
													 script5 = script5, div5 = div5, script6 = script6, div6 = div6, 
													 script7 = script7, div7 = div7,script8 = script8, div8 = div8)
		
    except Exception as e:
        return render_template('login.html')

def leyendas_plot(sensor_name):
# Esta función retorna el texto que irá en las leyendas de los gráficos según el nombre del sensor en la tabla cassandra
	
	if (sensor_name == "bq_7110_pt_1010"):
		return "Presión hidrostática real"
		
	elif (sensor_name == "bi_7110_dt_1030_solido"):
		return "Concentración sólidos descarga real"
		
	elif (sensor_name == "bo_7110_lt_1009_s4" ):
		return "Nivel de cama real"
		
	elif (sensor_name == "br_7120_ft_1002" ):
		return "Flujo Floculante real"
		
	elif (sensor_name == "bk_7110_ft_1030"):
		return "Flujo de descarga real"

####################################################
####################################################
@app.route('/', methods=['GET', 'POST'])
def home():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')
            
        else:
			# Se obtiene el nivel de usuario del usuario
            inject_data()
            
            # Se carga la página
            return index()
            
    except Exception as e:
        return render_template('login.html')

####################################################
####################################################
@app.route("/index")
def index():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')

        else:
			# Se obtiene el nivel de usuario del usuario
            inject_data()
            
			# Se carga la página
            return render_template("index.html")
            
    except Exception as e:
        return render_template('login.html')

@app.route("/Espesador_Prep")
def Espesador_Prep():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')

        else:
			# Se obtiene el nivel de usuario del usuario
            inject_data()
            
			# Se carga la página
            return render_template("Espesador_Prep.html")
            
    except Exception as e:
        return render_template('login.html')


####################################################
####################################################
@app.route("/deposito")
def home_dep():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')
            
        else:
			# Se obtiene el nivel de usuario del usuario
            inject_data()
            
            # Se carga la página
            return render_template('deposito.html')

    except Exception as e:
        return render_template('login.html')
        
####################################################
####################################################
@app.route("/tendencias_1" )
def home_tendencias():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')
            
        else:
			# Se obtiene el nivel de usuario del usuario

            inject_data()
            
            # Se carga la página
            return render_template('tendencias_1.html')
		
    except Exception as e:
        return render_template('login.html')

####################################################
####################################################
@app.route("/tendencias_2" )
def home_tendencias_2():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')
            
        else:
			# Se obtiene el nivel de usuario del usuario
            inject_data()
            
            # Se carga la página
            return render_template('tendencias_2.html')
		
    except Exception as e:
        return render_template('login.html')

####################################################
####################################################
@app.route("/tendencias_3" )
def home_tendencias_3():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')
            
        else:
			# Se obtiene el nivel de usuario del usuario
            inject_data()
            
            # Se carga la página
            return render_template('tendencias_3.html')
		
    except Exception as e:
        return render_template('login.html')
        
####################################################
####################################################
@app.route("/tendencias_4" )
def home_tendencias_4():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')
            
        else:
			# Se obtiene el nivel de usuario del usuario
            inject_data()
            
            # Se carga la página
            return render_template('tendencias_4.html')
		
    except Exception as e:
        return render_template('login.html')

####################################################
####################################
@app.route('/est-meteorologica')
def home_est():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')

        else:
			# Se obtiene el nivel de usuario del usuario
            inject_data()
            
            # Se carga la página
            return render_template('est_met.html')

    except Exception as e:
        return render_template('login.html')
        
####################################################
####################################################
@app.route('/control')
def home_control():
    try:
		
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
            return render_template('login.html')

        else:
			# Se obtiene el nivel de usuario del usuario
            inject_data()
            
            # Se carga la página
            return render_template('control.html')

    except Exception as e:
        return render_template('login.html')

####################################################
####################################################
@app.route('/login', methods=['GET', 'POST'])
def login_user():

    try:
		# Si el usuario no ha iniciado sesión se redirige hacia la página de login
        if (User.access_by_sessionid(session['uid']) > ACCESS['login']):
         return index()

    except Exception as e:
        pass		
	
    try:
		# Esta función comprueba si el usuario inició sesión, si no lo fuese el caso se redirige a la página de login
		
		# Se obtiene el nombre de usuario ingresado en la página de login	
        username = request.form['username']
        
        # Se obtiene la contraseña ingresada en la página de login
        password = request.form['password']

		# Se comprueba si los datos son correctos
        if User.is_login_valid(username, password):
			
			# Se da acceso al usuario con su nivel de acceso  
            user = User.find_by_username(username)
            session['uid'] = user.cookie
		
        return home()

    except Exception as e:
        return render_template('login.html')

####################################################
####################################################
@app.route("/logout", methods=['GET', 'POST'])
def logout():
# Esta función elimina el identificador del usuario para que ya no pueda acceder a las funcionalidades de la página

    session['uid'] = 0
    return home()
    
####################################################
####################################################
@app.context_processor
def inject_data():
# Esta función entrega un diccionario que incluye el nivel de acceso del usuario (access) y los niveles de acceso establecidos (ACCESS)

    try:
        return dict(access=User.access_by_sessionid(session['uid']), ACCESS=ACCESS)

    except Exception as e:
        return dict(access=0, ACCESS=ACCESS)

####################################################
####################################################
""" 
@app.before_request
def before_request():
# Esta función limita el tiempo que un usuario puede estar inactivo dentro de la sesión,
# si se cumple este límite se le cierra la sesión al usuario

    session.permanent = True
    app.permanent_session_lifetime = datetime.timedelta(seconds=10)
    session.modified = True
    
   
@app.route('/_check_session', methods=['POST'])
def check_session():
	
	try:
		render_template('login.html')
		if not (User.access_by_sessionid(session['uid']) > ACCESS['login']):
			return render_template('login.html')
		else:
			return render_template('login.html')

	except Exception as e:
		print(e)
		return jsonify(response= 0)        


scheduler = BackgroundScheduler()
scheduler.add_job(func=correr, trigger="interval", seconds=3)
scheduler.start()

    
atexit.register(lambda: scheduler.shutdown())
    # if __name__ == '__main__':
"""
if __name__ == '__main__':
    app.debug = True
    app.run(host='10.100.49.13', port=5000)


#app.run(debug=True)
