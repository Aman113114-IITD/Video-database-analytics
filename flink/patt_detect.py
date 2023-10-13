from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, DataTypes
from pyflink.table.sources import CsvTableSource
from pyflink.table import expressions as expr

# Create a StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)  # Set the parallelism for the job

# Create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# Define the schema for the input stream
field_names = ["Frameid", "Obj1", "Obj2", "direction"]
field_types = [DataTypes.STRING(), DataTypes.STRING(), DataTypes.STRING(), DataTypes.STRING()]

# Specify the schema and register the CSV table source
t_env.register_table_source(
    "InputData",
    CsvTableSource(
        'input_data.csv',field_names,field_types,field_delim=','
    )
)

# Define a pattern to detect the desired sequence
pattern_query = f"""
    SELECT A.*
    FROM InputData A, InputData B
    WHERE A.Obj1 = B.Obj1
      AND A.Obj2 = B.Obj2
      AND A.direction = 'N'
      AND B.direction = 'W'
"""

# Execute the pattern query
pattern_table = t_env.sql_query(pattern_query)

# Print the detected patterns
pattern_table.execute().print()
