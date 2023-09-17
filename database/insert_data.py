import snowflake.connector

# Replace these with your Snowflake connection details
snowflake_account = 'KFRKMFD-RN89837'
snowflake_user = 'vishwasbtp'
snowflake_password = 'Videoanalytics1234'
snowflake_database = 'BTP'
snowflake_schema = 'BTPDATA'

# Data to be inserted (assuming a list of tuples where each tuple represents a row)
data_to_insert = [
    (1, 101, 1, 'Red', 10.5, 20.5, 50.3, 70.1),
    (2, 102, 2, 'Blue', 15.2, 25.8, 55.6, 75.3),
    # Add more rows as needed
]

# SQL insert statement with parameterized placeholders
insert_query = f"""
    INSERT INTO VIDEO_LOGS
    (FRAME_ID, OBJECT_ID, OBJECT_LABEL, OBJECT_COLOR, XMIN, YMIN, XMAX, YMAX)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

try:
    # Establish a Snowflake connection
    conn = snowflake.connector.connect(
        user=snowflake_user,
        password=snowflake_password,
        account=snowflake_account,
        database=snowflake_database,
        schema=snowflake_schema
    )

    # Create a cursor
    cursor = conn.cursor()

    # Execute the insert statement using execute_many
    cursor.executemany(insert_query, data_to_insert)

    # Commit the transaction
    conn.commit()

    print("Data inserted successfully!")

except snowflake.connector.Error as e:
    print("Snowflake error:", e)

finally:
    # Close the cursor and connection
    cursor.close()
    conn.close()