import re
import torch
from typing import Dict, List, Tuple

import psycopg2
import pandas as pd

from pymilvus import model
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams
from pymilvus.model.reranker import CrossEncoderRerankFunction

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langchain_community.utilities import SQLDatabase
from urllib.parse import quote


class LLM_Model:
    def __init__(self, model_name_or_path, dtype="16"):
        is_enough_memory = torch.cuda.get_device_properties(0).total_memory > 15e9
        if not (torch.cuda.is_available() and is_enough_memory):
            raise Exception(
                "GPU is not available \
                            or does not have enough memory (16GB required)."
            )

        config = None

        if dtype == "8":
            config = BitsAndBytesConfig(
                load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif dtype == "4":
            config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=config,
            device_map="auto",
            use_cache=True,
        )

        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

    def create_prompt(
        self,
        user_question: str,
        instructions: str = "",
        create_table_statements: str = "",
        question_sql_pairs: str = "",
    ):
        prompt_template = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + "Generate a SQL query to answer this question: `{user_question}`\n\n"
            + "Use instructions below if needed\n"
            + "- If you cannot answer the question with the"
            + "available database schema, return 'I do not know.`\n"
            + "- If the question ask with string, Use where lower() LIKE '%%'\n"
            + "{instructions}\n\nDDL statements:\n{create_table_statements}\n\n"
            + "- Refer sample question-sql pairs below:\n" + "-" * 10 + "\n"
            + "{question_sql_pairs}\n"
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            + "The following SQL query best answers the question `{user_question}`:\n"
            + "```sql"
        )

        return prompt_template.format(
            user_question=user_question,
            instructions=instructions,
            create_table_statements=create_table_statements,
            question_sql_pairs=question_sql_pairs,
        )

    def submit_prompt(
        self,
        user_question: str,
        instructions: str = "",
        create_table_statements: str = "",
        question_sql_pairs: str = "",
    ):
        prompt = self.create_prompt(
            user_question=user_question,
            instructions=instructions,
            create_table_statements=create_table_statements,
            question_sql_pairs=question_sql_pairs,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            **inputs,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=400,
            do_sample=False,
            num_beams=1,
        )
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return prompt, outputs[0].split("```sql")[-1]


class MilvusDB_VectorStore:
    def __init__(
        self,
        db_name,
        dim=384,
        rerank_function: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_model: str = "all-MiniLM-L6-v2",
        device="cpu",
    ):
        self.client = MilvusClient(db_name)
        self.dim = dim
        self.embedding_model = model.dense.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model, device=device
        )
        self.rerank_function = CrossEncoderRerankFunction(
            model_name=rerank_function, device=device
        )

        self.doc_collection = "doc_collection"
        self.table_column_collection = "table_column_collection"
        self.ddl_collection = "ddl_collection"
        self.ddl_guide_collection = "ddl_guide_collection"
        self.qs_pair_collection = "qs_pair_collection"

    def create_index_params(
        self,
        vector_field: str = "vector",
        index_name: str = "",
        index_type: str = "",
        metric_type="COSINE",
    ):
        index_params = IndexParams()
        index_params.add_index(
            vector_field, index_type, index_name, metric_type=metric_type
        )
        return index_params

    def create_schema(self, fields: List[Dict]):
        schema = MilvusClient.create_schema(
            auto_id=True,
            primary_field="id",
            enable_dynamic_field=True,
        )
        [schema.add_field(**field) for field in fields]
        return schema

    def create_collection(
        self, collection_name, schema, index_params=None, metric_type="COSINE"
    ):
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.dim,
            schema=schema,
            index_params=index_params,
            metric_type=metric_type,
        )

    def create_table_column_collection(self):
        schema = self.create_schema(
            fields=[
                {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
                {
                    "field_name": "column_name",
                    "datatype": DataType.VARCHAR,
                    "max_length": 256,
                },
                {
                    "field_name": "vector",
                    "datatype": DataType.FLOAT_VECTOR,
                    "dim": self.dim,
                },
                {
                    "field_name": "table_name",
                    "datatype": DataType.VARCHAR,
                    "max_length": 512,
                },
            ]
        )

        index_params = self.create_index_params()

        try:
            self.create_collection(
                self.table_column_collection, schema, index_params=index_params
            )
        except Exception as ex:
            raise ex from ex

    def create_ddl_collection(self):
        schema = self.create_schema(
            fields=[
                {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
                {
                    "field_name": "table_name",
                    "datatype": DataType.VARCHAR,
                    "max_length": 512,
                },
                {
                    "field_name": "vector",
                    "datatype": DataType.FLOAT_VECTOR,
                    "dim": self.dim,
                },
                {
                    "field_name": "table_ddl",
                    "datatype": DataType.VARCHAR,
                    "max_length": 2**15,
                },
            ]
        )

        index_params = self.create_index_params()

        try:
            self.create_collection(
                self.ddl_collection, schema, index_params=index_params
            )
        except Exception as ex:
            raise ex from ex

    def create_doc_collection(self):
        schema = self.create_schema(
            fields=[
                {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
                {"field_name": "doc", "datatype": DataType.VARCHAR, "max_length": 512},
                {
                    "field_name": "vector",
                    "datatype": DataType.FLOAT_VECTOR,
                    "dim": self.dim,
                },
            ]
        )

        index_params = self.create_index_params()

        try:
            self.create_collection(
                self.doc_collection, schema, index_params=index_params
            )
        except Exception as ex:
            raise ex from ex

    def create_ddl_guide_collection(self):
        schema = self.create_schema(
            fields=[
                {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
                {
                    "field_name": "guide",
                    "datatype": DataType.VARCHAR,
                    "max_length": 256,
                },
                {
                    "field_name": "table_name",
                    "datatype": DataType.VARCHAR,
                    "max_length": 1024,
                },
                {
                    "field_name": "vector",
                    "datatype": DataType.FLOAT_VECTOR,
                    "dim": self.dim,
                },
            ]
        )

        index_params = self.create_index_params()

        try:
            self.create_collection(
                self.ddl_guide_collection, schema, index_params=index_params
            )
        except Exception as ex:
            raise ex from ex

    def create_question_sql_pair(self):
        schema = self.create_schema(
            fields=[
                {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
                {
                    "field_name": "question",
                    "datatype": DataType.VARCHAR,
                    "max_length": 512,
                },
                {"field_name": "sql", "datatype": DataType.VARCHAR, "max_length": 1024},
                {
                    "field_name": "vector",
                    "datatype": DataType.FLOAT_VECTOR,
                    "dim": self.dim,
                },
            ]
        )

        index_params = self.create_index_params()

        try:
            self.create_collection(
                self.qs_pair_collection, schema, index_params=index_params
            )
        except Exception as ex:
            raise ex from ex

    def start(self):
        self.create_table_column_collection()
        self.create_ddl_collection()
        self.create_doc_collection()
        self.create_ddl_guide_collection()
        self.create_question_sql_pair()

    @staticmethod
    def segment_sentence(sentence):
        from underthesea import pos_tag

        return ", ".join(
            [word for word, tag in pos_tag(sentence) if "V" in tag or "N" in tag or 'A' in tag]
        )

    @staticmethod
    def process_ddl(ddl: str):
        regex_statement = r"^\t(?!.*(?:CONSTRAINT|id|name|write_date|create_date)).*"
        column_lines = []
        
        if len(ddl) >= 5000:
            column_lines = re.findall(regex_statement, ddl, re.MULTILINE)
            column_lines = [
                column.replace("\t", "").replace(", ", "") for column in column_lines
            ]

        remove_constraint_regex = r"^\t(CONSTRAINT).*"
        modified_ddl = re.sub(remove_constraint_regex, "", ddl, flags=re.MULTILINE)
        modified_ddl = re.sub(regex_statement, "", modified_ddl, flags=re.MULTILINE)
        modified_ddl = re.sub(r"\n\s*\n", "\n", modified_ddl).strip()

        return modified_ddl, column_lines

    def insert_into_collection(self, collection_name, data: List[Dict]):
        return self.client.insert(collection_name, data)

    def insert_table_columns(self, table_columns: List[Tuple[str, List[str]]]):
        data = []
        for table_name, columns in table_columns:
            data.extend(
                [
                    {
                        "table_name": table_name,
                        "vector": self.embedding_model.encode_documents([col])[0],
                        "column_name": col,
                    }
                    for col in columns
                ]
            )

        return self.insert_into_collection(self.table_column_collection, data)

    def insert_ddl_statements(self, ddls: List[Tuple[str, str]]):
        data = [
            {
                "table_name": name,
                "vector": self.embedding_model.encode_documents([name])[0],
                "table_ddl": ddl,
            }
            for name, ddl in ddls
        ]

        return self.insert_into_collection(self.ddl_collection, data)

    def insert_docs(self, docs: List[str]):
        data = [
            {
                "doc": doc,
                "vector": self.embedding_model.encode_documents(
                    [self.segment_sentence(doc)]
                )[0],
            }
            for doc in docs
        ]

        return self.insert_into_collection(self.doc_collection, data)

    def insert_question_sql_pair(self, list_pairs: List[Tuple[str, str]]):
        data = [
            {
                "question": question,
                "sql": sql,
                "vector": self.embedding_model.encode_documents([question])[0],
            }
            for question, sql in list_pairs
        ]

        return self.insert_into_collection(self.qs_pair_collection, data)

    def insert_ddl_guides(self, guides: List[Tuple[str, List]]):
        data = [
            {
                "guide": guide,
                "table_name": ", ".join(tables),
                "vector": self.embedding_model.encode_documents([guide])[0],
            }
            for guide, tables in guides
        ]

        return self.insert_into_collection(self.ddl_guide_collection, data)

    @staticmethod
    def _extract_query_results(results):
        if len(results) == 0:
            return ""

        if isinstance(results[0], tuple):
            res = ""
            for q, sql in results:
                res += f"Question `{q}`:\n"
                res += f"```sql\n{sql}\n"
                res += "-" * 10 + "\n"

            return res

        return "\n".join(results)

    def get_related_table_columns(self, table_name, question, n_cols=5):
        ann_search = self.client.search(
            collection_name=self.table_column_collection,
            anns_field="vector",
            data=self.embedding_model.encode_documents([question]),
            limit=n_cols,
            output_fields=["column_name"],
            filter=f"table_name == '{table_name}'",
        )

        return [doc["entity"]["column_name"] for doc in ann_search[0]]

    def get_many_related_table_columns(
        self, table_name, questions: List[str], n_cols=5
    ) -> List:
        columns_return = []
        for question in questions:
            columns = self.get_related_table_columns(table_name, question, n_cols)
            columns_return.extend(columns)

        return list(set(columns_return))

    def get_related_ddl_guides(self, question, n_guides=5) -> List:
        ann_search = self.client.search(
            collection_name=self.ddl_guide_collection,
            anns_field="vector",
            data=self.embedding_model.encode_documents([question]),
            limit=n_guides,
            output_fields=["guide", "table_name"],
        )

        ann_search = {
            doc["entity"]["guide"]: doc["entity"]["table_name"] for doc in ann_search[0]
        }

        reranked_results = self.rerank_function(
            query=question, documents=list(ann_search.keys()), top_k=2
        )

        create_statements = [
            ann_search[key.text].split(", ") for key in reranked_results
        ]
        create_statements = [
            statement for statements in create_statements for statement in statements
        ]

        return list(set(create_statements))

    def get_related_docs(self, question, n_docs=5) -> str:
        ann_search = self.client.search(
            collection_name=self.doc_collection,
            anns_field="vector",
            data=self.embedding_model.encode_documents([question]),
            limit=n_docs,
            output_fields=["doc"],
        )

        return self._extract_query_results(
            [doc["entity"]["doc"] for doc in ann_search[0]]
        )

    @staticmethod
    def add_columns_to_ddl(ddl: str, columns: List[str]):
        tab_index = ddl.find("\t")
        for col in columns:
            ddl = ddl[:tab_index] + f"\t{col}, \n" + ddl[tab_index:]
        return ddl + ";\n"

    def get_related_ddls(self, question, docs: str, n_ddls=1, n_docs=5, n_cols=5) -> List:
        ann_search = self.client.search(
            collection_name=self.ddl_collection,
            anns_field="vector",
            data=self.embedding_model.encode_documents([question]),
            limit=n_ddls,
            output_fields=["table_ddl", "table_name"],
        )
        table_ddl = [doc["entity"]["table_ddl"] for doc in ann_search[0]]
        table_name = [doc["entity"]["table_name"] for doc in ann_search[0]]
        docs = docs.split("\n")

        for i, name in enumerate(table_name):
            columns = self.get_many_related_table_columns(name, docs, n_cols)
            table_ddl[i] = self.add_columns_to_ddl(table_ddl[i], columns)

        return table_ddl

    def get_many_related_ddls(
        self, list_guides: List[str], docs: str, n_ddls=1, n_docs=5, n_cols=5
    ) -> str:
        related_ddls = []
        for question in list_guides:
            ddls = self.get_related_ddls(question, docs, n_ddls, n_docs, n_cols)
            related_ddls.extend(ddls)

        return self._extract_query_results(list(set(related_ddls)))

    def get_related_question_sql_pair(self, question, n_pair=5):
        search_vector = self.embedding_model.encode_documents([question])

        ann_search = self.client.search(
            collection_name=self.qs_pair_collection,
            anns_field="vector",
            data=search_vector,
            limit=n_pair,
            output_fields=["sql", "question"],
        )

        related_pairs = [
            (pair["entity"]["question"], pair["entity"]["sql"])
            for pair in ann_search[0]
        ]

        return self._extract_query_results(related_pairs)

    def get_row_by_ids(self, ids: List, collection_name):
        return self.client.query(
            collection_name=collection_name,
            ids=ids,
        )

    def compute_cosine_distance(self, s1, s2):
        from scipy.spatial import distance

        s1 = self.embedding_model.encode_documents([s1])[0]
        s2 = self.embedding_model.encode_documents([s2])[0]
        print(distance.cosine(s1, s2))


class Rag2SQL_Model(MilvusDB_VectorStore, LLM_Model):
    def __init__(self, db_name, model_name_or_path, dtype):
        MilvusDB_VectorStore.__init__(self, db_name)
        LLM_Model.__init__(self, model_name_or_path, dtype)
        self.connection = None
        self.connection_string = ""

    def connect_to_postgres(
        self,
        host: str,
        dbname: str,
        user: str,
        password: str,
        port: int,
    ):
        try:
            self.connection = psycopg2.connect(
                database=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
            )

            user_encoded = quote(user)
            password_encoded = quote(password)

            self.connection_string = (
                f"postgresql://{user_encoded}:{password_encoded}@{host}:{port}/{dbname}"
            )

        except Exception as e:
            print(f"The error '{e}' occurred")

    def prepare_ddls(self):
        db = SQLDatabase.from_uri(
            self.connection_string,
            sample_rows_in_table_info=0,
            view_support=True,
        )

        ddls = db.get_context()["table_info"].split("\n\n")
        mod_ddls = []
        table_columns = []

        for ddl in ddls:
            table_name = re.search(r"CREATE TABLE (\w+) \(", ddl).group(1)
            new_ddl, cols = self.process_ddl(ddl)
            mod_ddls.append((table_name, new_ddl))
            table_columns.append((table_name, cols))

        return mod_ddls, table_columns

    def run_sql(self, query):
        if self.connection:
            cursor = self.connection.cursor()
            try:
                cursor.execute(query)
                results = cursor.fetchall()

                df = pd.DataFrame(
                    results, columns=[desc[0] for desc in cursor.description]
                )
                return df

            except Exception as e:
                self.connection.rollback()
                print(f"The error '{e}' occurred")
                raise Exception("Error when query: ", e)

    @staticmethod
    def _extract_sql(sql_response):
        # return sql_response

        sensitive_keywords = [
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "INSERT",
            "UPDATE",
        ]

        for keyword in sensitive_keywords:
            if keyword in sql_response.upper():
                return "SELECT 'Tôi không biết' as answer;"

        sqls = re.findall(r"\bWITH\b .*?;", sql_response, re.DOTALL)
        if sqls:
            return sqls[-1]

        sqls = re.findall(r"SELECT.*?;", sql_response, re.DOTALL)
        if sqls:
            return sqls[-1]

        return "SELECT 'Đã có lỗi xảy ra!' as answer;"

    def generate_query(self, question):
        sumarize_question = self.segment_sentence(question)
        docs = self.get_related_docs(sumarize_question)
        guides = self.get_related_ddl_guides(sumarize_question)
        ddls = self.get_many_related_ddls(guides, docs)
        question_sql_pair = self.get_related_question_sql_pair(sumarize_question)

        prompt, query = self.submit_prompt(question, docs, ddls, question_sql_pair)

        return prompt, self._extract_sql(query)

    def train(
        self,
        ddls: List[Tuple[str, str]] = None,
        columns: List[Tuple[str, List[str]]] = None,
        guides: List[Tuple[str, List]] = None,
        docs: List[str] = None,
        question_sql_pairs: List[Tuple[str, str]] = None,
    ):
        self.start()

        if columns:
            print("Columns: ", self.insert_table_columns(columns))
        if ddls:
            print("DDLs: ", self.insert_ddl_statements(ddls))
        if guides:
            print("Guides: ", self.insert_ddl_guides(guides))
        if docs:
            print("Docs: ", self.insert_docs(docs))
        if question_sql_pairs:
            print("Q&S Pair: ", self.insert_question_sql_pair(question_sql_pairs))

    def ask(self, question):
        try:
            prompt, sql = self.generate_query(question)
        except Exception as e:
            print("Error when generate sql: ", e)
            return "Tôi chịu", None, None

        try:
            res = self.run_sql(sql)
            return res, prompt, sql
        except Exception as e:
            print("Error when run sql: ", e)
            return "Đang có trục trặc gì ấy.. .-.", prompt, sql


if __name__ == "__main__":
    guide_docs = [
        ("phòng, department", ["hr_department"]),
        ("bệnh nhân, patient", ["medical_patient"]),
        ("staff, employee", ["hr_employee"]),
        ("sản phẩm, product", ["product_product"]),
        ("phôi, embryo", ["medical_embryo_culture"]),
        (
            "xét nghiệm, kết quả, chỉ số, test result, test indices, E2 index",
            [
                "medical_test",
                "medical_test_result",
                "medical_test_indices",
            ],
        ),
        (
            "đơn thuốc, toa thuốc, prescription",
            [
                "medical_prescription_order",
                "medical_prescription_order_line",
            ],
        ),
        (
            "điều trị thành công, successful treatment, kết quả điều trị, treatment result, beta dương tính, positive beta, IVF",
            [
                "medical_treatment",
                "medical_treatment_result",
                "medical_treatment_result_beta",
            ],
        ),
        (
            "mẫu trữ, cryopreserv straw, trữ tinh trùng, sperm cryopreserv.",
            [
                "medical_cryopreserv_straw",
                "medical_storage_process",
            ],
        ),
    ]
    docs = [
        "- giới tính, nam, nữ ~ gender",
        "- tuổi ~ birthday",
        "- địa điểm/nơi cưới ~ marriage_registration_place",
        "- điều trị thành công ~ num_baby_live_digit > 0",
        "- năm điều trị ngày khám đầu ~ date_first",
        "- mẫu trữ hết hạn ~ date_expired",
        "- kết quả beta dương tính ~ conclude = 'positive'",
        "- mẫu trữ tinh trùng ~ type_cryopreserv = 'sperm'",
        "- thể tích trước trữ ~ volume_before_stored_sperm",
        "- phôi có kiểu hình, ngày 0 ~ day_info_0",
        "- chỉ số E2 ~ medical_test_indices.name = 'E2'",
        "- điều trị chỉ là IVF ~ treatment_type_ids.selection.value",
    ]
    question_sql_pair = [
        (
            "có bao nhiêu bệnh nhân họ Nguyễn",
            "SELECT first_name FROM medical_patient mp WHERE unaccent(mp.first_name) ILIKE unaccent('nguyen%');",
        ),
        (
            "Có bao nhiêu bệnh nhân lớn hơn 30 tuổi (Trung niên)",
            "SELECT COUNT (*) FROM medical_patient mp WHERE EXTRACT (YEAR FROM AGE (CURRENT_DATE, mp.birthday)) > 30;",
        ),
        (
            "Liệt kê bệnh nhân điều trị thành công trong năm 2023",
            """SELECT mp.name,
       mp.date_first,
FROM medical_patient mp
JOIN medical_treatment_result mt ON mp.id = mt.patient_id
WHERE mt.num_baby_live_digit > 0
  AND EXTRACT (YEAR
               FROM mp.date_first) = 2023;""",
        ),
        (
            "Bệnh nhân nào có mẫu trữ sắp hết hạn trong 30 ngày tới",
            """SELECT DISTINCT mcs.date_expired, mp.name
FROM medical_cryopreserv_straw mcs
JOIN medical_patient mp ON mcs.patient_id = mp.id
WHERE mcs.date_expired BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
ORDER BY mp.name""",
        ),
        (
            "Liệt kê bệnh nhân có kết quả điều trị có kết quả beta dương tính",
            """SELECT mp.name
FROM medical_patient mp
JOIN medical_treatment_result tr ON mp.id = tr.patient_id
JOIN medical_treatment_result_beta mtrb ON tr.id = mtrb.result_id
WHERE tr.conclude = 'positive';""",
        ),
        (
            "Có bao nhiêu mẫu trữ tinh trùng có thể tích trước trữ > 1.5 ml",
            """SELECT COUNT (*)
FROM medical_storage_process msp
WHERE msp.volume_before_stored_sperm > 1.5;""",
        ),
        (
            "Những phôi có kiểu hình ngày 0 là MII",
            """SELECT COUNT(*)
FROM medical_embryo_culture me
JOIN medical_embryo_culture me2 on me.id = me2.day_info_0
WHERE me.name ILIKE '%mii%'""",
        ),
        (
            "Bao nhiêu xét nghiệm có chỉ số E2 > 20",
            """SELECT COUNT(*)
FROM medical_test_result mtr
JOIN medical_test mt on mt.id = mtr.test_id
JOIN medical_test_indices mti on mti.id = mtr.test_indices_id
WHERE mti.name::json->>'en_US' ILIKE '%e2%'
AND (
    CASE
        WHEN mtr.result::json->>'value' ~ '^[0-9]+(\\.[0-9]+)?$' THEN (mtr.result::json->>'value')::NUMERIC
        ELSE NULL
    END
) > 20;""",
        ),
    ]

    r = MilvusDB_VectorStore("milvus_demo.db")

    if False:
        postgres_uri = (
            "postgresql://ims_ro:imsro%406F17A4E0@14.224.150.150:54321/hp_migrate"
        )

        db = SQLDatabase.from_uri(
            postgres_uri,
            sample_rows_in_table_info=0,
            view_support=True,
        )

        ddls = db.get_context()["table_info"].split("\n\n")
        mod_ddls = []
        table_columns = []

        for ddl in ddls:
            table_name = re.search(r"CREATE TABLE (\w+) \(", ddl).group(1)
            new_ddl, cols = r.process_ddl(ddl)
            mod_ddls.append((table_name, new_ddl))
            table_columns.append((table_name, cols))

        # r.create_table_column_collection()
        # r.create_ddl_collection()
        # r.create_doc_collection()
        # r.create_ddl_guide_collection()
        # r.create_question_sql_pair()
        r.start()
        r.insert_ddl_statements(mod_ddls)
        r.insert_table_columns(table_columns)
        r.insert_ddl_guides(guide_docs)
        r.insert_docs(docs)
        r.insert_question_sql_pair(question_sql_pair)

    else:
        from underthesea import pos_tag

        while True:
            q = input("Enter: ")

            if q == "c":
                import os

                os.system("clear")
                continue

            q = ", ".join(
                [word for word, tag in pos_tag(q) if "V" in tag or "N" in tag or 'A' in tag]
            )

            print(q)
            guides = r.get_related_ddl_guides(q)
            r_docs = r.get_related_docs(q)

            # print(guides)
            print(r.get_many_related_ddls(guides, r_docs))
            print("-" * 20)
            print(r_docs)
            print("-" * 20)
            print(r.get_related_question_sql_pair(q))

            # print(r.get_related_docs("có bao nhiêu xét nghiệm?"))
            # print(r.get_related_question_sql_pair('có bao nhiêu xét nghiệm?'))

            # r.get_related_table_columns('medical_test', 'xét nghiệm')
            r.compute_cosine_distance(
                "Có, bệnh nhân, có, xét nghiệm?",
                "người xác thực xét nghiệm = validate_user_id",
            )
            r.compute_cosine_distance(
                "nghề nghiệp = occupation",
                "Có, bệnh nhân, có, xét nghiệm?",
            )

            q = 2  # for debug
