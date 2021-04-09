from cassandra.cluster import Cluster
from cassandra import timestamps
from werkzeug.security import generate_password_hash
import uuid

cluster = Cluster()
keyspace = 'usuarios'
connection = cluster.connect(keyspace)

Names = ["admin","admin1", "desarrollador", "operador"]
Pass = ["admin","admin", "dev", "op"]
Access = [3,3,2,1]



for i in range(0, len(Names)):
 connection.execute("""
     INSERT INTO Usuarios.data (name, password, access, session_id)
     VALUES (%(name)s, %(password)s, %(access)s, %(session_id)s)
     """,
     {'name':Names[i]  , 'password':generate_password_hash(Pass[i], method='sha256'), 'access':Access[i], 'session_id': uuid.uuid4()})

