from db import get_connection

# Run schema once
with open("db/schema.sql", "r") as file:
    schema = file.read()

conn = get_connection()
cur = conn.cursor()
cur.execute(schema)
conn.commit()
cur.close()
conn.close()

print("Schema created.")