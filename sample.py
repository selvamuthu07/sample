import streamlit as st
import pandas as pd
import duckdb
import openai
import json
from streamlit_echarts import st_echarts

key1 = "sk-proj-QCJ8MQxdUiUV3xR1VteJOhFzEzIc8"
key2 = "53ghYiquB8qC0vhqtjoiV4wBMDuNDxVzR2L-Qmpx_v4RzT3BlbkFJw07ud"
key3 =  "OI2vDDz70SExXFYdWZeHNG8GIwqgXYAloxQZpLQUg6zLz9c-KZVIyDqUvcz1b7yD8wyQA"

openai.api_key = key1 + key2 + key3
 
DB_NAME = ':memory:'
TABLE_NAME = 'data'
 
def table_exists(conn, TABLE_NAME):
    try:
        result = conn.execute(f"SELECT * FROM information_schema.tables WHERE table_name='{TABLE_NAME}'").fetchdf()
        return not result.empty
    except Exception:
        return False
 
def get_table_columns(conn, TABLE_NAME):
    try:
        schema = conn.execute(f"DESCRIBE {TABLE_NAME}").fetchdf()
        return [col.lower() for col in schema['column_name'].tolist()]
    except Exception:
        return []
 
if 'duckdb_conn' not in st.session_state:
    st.session_state['duckdb_conn'] = duckdb.connect(database=DB_NAME)
if 'csv_file_uploaded' not in st.session_state:
    st.session_state['csv_file_uploaded'] = False
if 'df_clean' not in st.session_state:
    st.session_state['df_clean'] = None
if 'schema' not in st.session_state:
    st.session_state['schema'] = None
if 'schema_description' not in st.session_state:
    st.session_state['schema_description'] = None
if 'unique_column_values' not in st.session_state:
    st.session_state['unique_column_values'] = None
if 'user_query' not in st.session_state:
    st.session_state['user_query'] = None
if 'required_columns' not in st.session_state:
    st.session_state['required_columns'] = False
if 'sql_query' not in st.session_state:
    st.session_state['sql_query'] = None
if 'sql_query_sucess' not in st.session_state:
    st.session_state['sql_query_sucess'] = False
if 'required_data_retrival' not in st.session_state:
    st.session_state['required_data_retrival'] = False
if 'df_result' not in st.session_state:
    st.session_state['df_result'] = None
if 'visual_code' not in st.session_state:
    st.session_state['visual_code'] = None
if 'visual_code_generarted' not in st.session_state:
    st.session_state['visual_code_generarted'] = False
if 'error' not in st.session_state:
    st.session_state['error'] = False

st.session_state['meta_data'] = None
st.session_state['meta_data1'] = None
 
conn = st.session_state['duckdb_conn']

def generate_column_metadata(df: pd.DataFrame):
    metadata = {
        "table_name": TABLE_NAME,
        "overall_row_count" : len(df),
        "columns": []
    }

    for col in df.columns:
        col_data = df[col]
        col_info = {}
        col_info['column_name'] = col
        col_info['data_type'] = str(col_data.dtype)

        if pd.api.types.is_datetime64_any_dtype(col_data):
            semantic_type = "datetime"
        elif 'id' in col.lower():
            semantic_type = "identifier"
        elif pd.api.types.is_numeric_dtype(col_data):
            semantic_type = "measure"
        else:
            semantic_type = "dimension"

        col_info['semantic_type'] = semantic_type
        col_info['null_percent'] = round(col_data.isnull().mean() * 100, 2)

        unique_count = col_data.nunique(dropna=True)
        col_info['unique_count'] = int(unique_count)
        col_info['sample_values'] = col_data.dropna().unique()[:10].tolist()

        col_info['is_identifier'] = unique_count == len(col_data.dropna())
        col_info['is_numeric'] = pd.api.types.is_numeric_dtype(col_data)
        col_info['is_datetime'] = pd.api.types.is_datetime64_any_dtype(col_data)

        if col_info['is_datetime']:
            col_info["datetime_format"] = str(col_data.dropna().iloc[0])[:10]
        else:
            col_info["datetime_format"] = None

        col_info['is_categorical'] = unique_count < 20
        col_info['is_groupable'] = unique_count < (0.5 * len(col_data.dropna()))
        
        if col_info['is_numeric'] or col_info['is_datetime']:
            col_info['min'] = str(col_data.min())
            col_info['max'] = str(col_data.max())
        else:
            col_info['min'] = None
            col_info['max'] = None

        metadata["columns"].append(col_info)

    return metadata
 
with st.sidebar:
    st.title("📁 Upload CSV File")
    csv_file = st.file_uploader("Choose a CSV file", type=["csv"])
 
    if csv_file is not None and not st.session_state['csv_file_uploaded']:
        encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                df_clean = pd.read_csv(csv_file, encoding=encoding)
                break
            except Exception:
                pass
 
        df_clean.columns = [col.lower().strip().replace(" ", "_").replace("-", "_") for col in df_clean.columns]
        df_clean = df_clean.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
        df_clean.drop_duplicates(inplace=True)
        df_clean.dropna(how='all', axis=1, inplace=True)

        meta_data = generate_column_metadata(df_clean)
        st.session_state['meta_data'] = meta_data

        conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        conn.register('df_clean', df_clean)
        conn.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM df_clean")
        conn.unregister('df_clean')
 
        schema = conn.execute(f"DESCRIBE {TABLE_NAME}").fetchdf()
        st.markdown("**Table Schema:**")
        st.dataframe(schema[["column_name", "column_type"]], width=500, height=500)
        st.session_state['csv_file_uploaded'] = True
        st.session_state['df_clean'] = df_clean
        st.session_state['schema'] = schema
        
        st.session_state['schema_description'] = None
        st.session_state['unique_column_values'] = None
        st.session_state['user_query'] = None
        st.session_state['required_columns'] = False
        st.session_state['sql_query'] = None
        st.session_state['sql_query_sucess'] = False
        st.session_state['required_data_retrival'] = False
        st.session_state['df_result'] = None
        st.session_state['visual_code'] = None
        st.session_state['visual_code_generarted'] = False
        st.session_state['error'] = False
 
st.title("Chat to Chart Visualization")
 
if st.session_state['csv_file_uploaded'] and table_exists(conn, TABLE_NAME):
    st.dataframe(st.session_state['df_clean'].head(5))
    user_query = st.text_input("Ask a question about the data to generate visual")
    if user_query:
        st.session_state['user_query'] = user_query
 
    if st.button(" Generate 🚀") and st.session_state['user_query']:
        # 1. Generate schema description if not already done
        if not st.session_state.get("schema_description"):
            with st.spinner("Generating schema description..."):
                df_sample = st.session_state['df_clean'].head(3)
                df_sample = df_sample.to_dict(orient='records')
                prompt = f"""You are a data analyst. understand the sample json dataframe and its schema.
                Provide column descriptions.  
                sample  json dataframe: {df_sample}
 
            Output Format: json format {{"Column" :  "Description" }}."""
                response = openai.ChatCompletion.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                response = response['choices'][0]['message']['content']
                response = response.lstrip('```json').rstrip('```')
                st.session_state["schema_description"] = response
 
        with st.expander("GEN AI generated data column description", expanded=False):
            schema_json = json.loads(st.session_state["schema_description"])
            schema_df = pd.DataFrame(list(schema_json.items()), columns=["Column_name", "Description"])
            st.dataframe(schema_df)
 
        # 2. Get required columns for the user query
        with st.spinner("GEN AI Analyzing Required Columns..."):
            if not st.session_state.get("required_columns") or st.session_state["error"]:
                prompt = [
                    {"role": "user", "content": f"""Task : you are a data analyst, tasked to provide required column names from data schema only based on user query.
 
                        schema and column descriptions: {schema_df.to_dict(orient='records')}

                        metadata : {st.session_state['meta_data']}
 
                        User query: {st.session_state['user_query']}
 
                        Output format: list format ["Column_name1", "Column_name2", "Column_name3", ...]
 
                        Only return a column names which is case sensitive based on the schema and descriptions above.
            Do not manipulate the schema based on the user query. Understand context from user query and schema provide column names !!!"""}]
                response = openai.ChatCompletion.create(
                    model="gpt-4.1-mini",
                    messages=prompt,
                    temperature=0
                )
 
                response = response['choices'][0]['message']['content']
 
                try:
                    column_names = json.loads(response)
                    unique_column_values = {}
                    table_columns = get_table_columns(conn, TABLE_NAME)
                    for column in column_names:
                        if column.lower() in table_columns:
                            try:
                                df_col = conn.execute(f"SELECT DISTINCT {column} FROM {TABLE_NAME}").fetchdf()
                                values_list = df_col[column].tolist()
                                if len(values_list) > 20:
                                    values_list = values_list[:20] + ['...etc still more values']
                                unique_column_values[column] = ' | '.join(values_list)
                            except Exception as e:
                                unique_column_values[column] = f"Error: {e}"
                        else:
                            unique_column_values[column] = "Column not found in table!"
 
                    st.session_state["unique_column_values"] = unique_column_values
                    st.session_state["required_columns"] = True
                    st.session_state["error"] = False
 
                except json.JSONDecodeError as e:
                    st.write(f"⚠️ error occured: {e} 😵‍💫 — TRY AGAIN 🔁 or hit that UNIVERSAL TROUBLESHOOTING RESTART 🔧🚀")
                    st.session_state["error"] = True
                    st.stop()
 
        # 3. Generate SQL query
        with st.expander("GEN AI SQL Query Generation...", expanded=False):
            if st.session_state["required_columns"]:
                sample_df = st.session_state['df_clean'].head(5)
                prompt = [{"role": "user", "content": f"""Task : you are a data analyst, tasked to write SQL query to get data from DB duckdb based on user query.
 
                            Note : please write clear structred sql query by understanding
                                1) sample data of 5 rows
                                2) schema and column descriptions
                                3) user query
                                4) based on user query i will provide some of the column with sample values in data, you can use if it is really required or just understand synatax of data.
                                5) all data values and column names are in lower case and in string data type ensure while writing sql query.
 
                            DB duckdb table name: {TABLE_NAME}
 
                            sample data: {sample_df.to_dict(orient='records')}
 
                            schema and column descriptions: {schema_df.to_dict(orient='records')} 

                            metadata : {st.session_state['meta_data']}
 
                            User query: {st.session_state['user_query']}
 
                            some of the column values: {st.session_state['unique_column_values']}
 
                            Only return sql query with valid column names based on the schema and descriptions above.
                Do not manipulate the column names on your own. column name should be case sensitive !!!.
 
                Provide sql query only don't provide any unnessasary information.
 
                note : you are always making mistake in column zolgs_prescriber should be zolg_prescriber"""}]
 
                response = openai.ChatCompletion.create(
                    model="gpt-4.1",
                    messages=prompt,
                    temperature=0
                )
                response = response['choices'][0]['message']['content']
                response = response.lstrip('```sql').rstrip('```')
                st.code(response, language='sql')
                st.session_state["sql_query"] = response
                st.session_state["sql_query_sucess"] = True
 
        # 4. Run SQL and show result
        with st.expander("Required data filter using SQL query...", expanded=False):
            if st.session_state["sql_query_sucess"]:
                sql_query = st.session_state["sql_query"]
                if not table_exists(conn, TABLE_NAME):
                    st.write("⚠️ SQL error: Table does not exist! Please upload a CSV file and try again.")
                    st.session_state["required_data_retrival"] = False
                else:
                    try:
                        df_result = conn.execute(sql_query).fetchdf()
                        columns = df_result.columns.tolist()
                        st.dataframe(df_result)
                        st.session_state["required_data_retrival"] = True
                        st.session_state["df_result"] = df_result
                    except Exception as e:
                        st.write(f"⚠️ SQL error: {e}")
                        st.session_state["required_data_retrival"] = False
 
        # 5. Visualization code generation
        with st.expander("GEN AI Visulization code Generation..", expanded=False):
            if st.session_state["required_data_retrival"]:
                df_result = st.session_state["df_result"]
                meta_data1 = generate_column_metadata(df_result)
                st.session_state['meta_data1'] = meta_data1
                prompt = f"""
You are a data visualization expert.
 
You are given:
 
A pandas DataFrame named df (already filtered from a SQL database).
The list of column names in the DataFrame: {df_result.columns.tolist()}

metadata : {st.session_state['meta_data1']}

A user request for a visualization:
{st.session_state['user_query']}
Your job:
 
Write valid Python code that:
Preprocesses the DataFrame as needed for the visualization.
Creates a Python dictionary named option containing the ECharts configuration.
Ensures that all values in the option dictionary are JSON-serializable (i.e., only use Python lists, dicts, numbers, and strings; do not use pandas Series or DataFrames).
Uses st_echarts(options=option, height="500px") to display the chart in Streamlit.
Only use columns present in the DataFrame.
If the request cannot be fulfilled due to missing columns, output a comment in JSON: {{"error": "reason"}}
Only output the complete Python code (no explanations, no markdown).
Data sample (first 10 rows): {df_result.head(10).to_string(index=False)}
 
Key addition:
 
Ensures that all values in the option dictionary are JSON-serializable (i.e., only use Python lists, dicts, numbers, and strings; do not use pandas Series or DataFrames).
"""
                response = openai.ChatCompletion.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": prompt}]
                )
                response = response['choices'][0]['message']['content']
                response = response.lstrip('```json').rstrip('```')
                response = response.lstrip('```python').rstrip('```')
                response = response.lstrip('python').rstrip('```')
                st.code(response, language='echarts')
                st.session_state["visual_code_generarted"] = True
                st.session_state["visual_code"] = response
 
        # 6. Visualization rendering
        with st.spinner("Visualizing data..."):
            if st.session_state["required_data_retrival"] and st.session_state.get("visual_code"):
                try:
                    df = st.session_state["df_result"]
                    exec(st.session_state["visual_code"])
                except Exception as e:
                    st.write(f"⚠️ error occured: {e} 😵‍💫 — TRY AGAIN 🔁 or hit that UNIVERSAL TROUBLESHOOTING RESTART 🔧🚀")
else:
    if not st.session_state['csv_file_uploaded']:
        st.info("Please upload a CSV file to get started.")