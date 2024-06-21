import json

def parse_sql_file(file_path):
    list_sql = []
    query = ""

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if "CREATE TABLE" in line or "CREATE VIEW" in line:
                if query:
                    list_sql.append(query.strip())
                    query = ""
                query = line
            else:
                if query:
                    query += line

            if line.strip() == "" and query:
                list_sql.append(query.strip())
                query = ""

        if query:
            list_sql.append(query.strip())

    def split_long_queries(list_sql):
        result = []
        for query in list_sql:
            if len(query) > 4000:
                prefix = query.split('\n')[0] + '\n'
                suffix = '\n);\n\n'
                sub_queries = [prefix + query[i:i+2000] + suffix for i in range(0, len(query), 2000)]
                result.append(sub_queries)
            else:
                result.append([query])
        return result
    
    return split_long_queries(list_sql)

def add_ddl_to_json(list_ddl):
    table_info = {}

    for ddl in list_ddl:
        table_name = ddl.split('\n')[0]
        table_info[table_name] = ddl

    json_data = json.dumps(table_info, indent=4)
    
    with open("ddl_statements.json", "w", encoding="utf-8") as file:
        file.write(json_data)

    return json_data

if __name__ == "__main__":

    ddl = parse_sql_file("./ddl_statement/table_info.txt")
    print(ddl[0])