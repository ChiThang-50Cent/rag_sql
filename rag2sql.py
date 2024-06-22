import torch
from typing import Dict, List, Tuple

import psycopg2
import pandas as pd

from pymilvus import model
from pymilvus import MilvusClient, DataType
from pymilvus.model.reranker import CrossEncoderRerankFunction

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.utilities import SQLDatabase
from urllib.parse import quote

class LLM_Model:
    def __init__(self, model_name_or_path):
        is_enough_memory = torch.cuda.get_device_properties(0).total_memory > 15e9
        if not (torch.cuda.is_available() and is_enough_memory):
            raise Exception(
                "GPU is not available \
                            or does not have enough memory (16GB required)."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
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
            + "- If you cannot answer the question with the \
                available database schema, return 'I do not know.`\n"
            + "{instructions}\n\nDDL statements:\n{create_table_statements}\n\n"
            + "-- Refer some samples below:\n{question_sql_pairs}\n\n"
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

        return outputs[0].split("```sql")[-1]


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
        self.ddl_collection = "ddl_collection"
        self.ddl_guide_collection = "ddl_guide_collection"
        self.qs_pair_collection = "qs_pair_collection"

    def create_collection(self, collection_name, schema):
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.dim,
            metric_type="IP",
            schema=schema,
        )

    def create_ddl_collection(self):
        schema = MilvusClient.create_schema(auto_id=True, primary_field="id")

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="table_name", datatype=DataType.VARCHAR, max_length=512
        )
        schema.add_field(
            field_name="name_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(
            field_name="table_ddl", datatype=DataType.VARCHAR, max_length=2**15
        )

        self.create_collection(self.ddl_collection, schema)

        index_params = self.client.prepare_index_params()
        index_params.add_index("name_vector", "", "", metric_type="COSINE")
        self.client.create_index(self.ddl_collection, index_params)

    def create_doc_collection(self):
        schema = MilvusClient.create_schema(auto_id=True, primary_field="id")

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(
            field_name="doc_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )

        self.create_collection(self.doc_collection, schema)

        index_params = self.client.prepare_index_params()
        index_params.add_index("doc_vector", "", "", metric_type="COSINE")
        self.client.create_index(self.doc_collection, index_params)

    def create_ddl_guide_collection(self):
        schema = MilvusClient.create_schema(auto_id=True, primary_field="id")

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="guide", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(
            field_name="table_names", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="guide_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )

        self.create_collection(self.ddl_guide_collection, schema)

        index_params = self.client.prepare_index_params()
        index_params.add_index("guide_vector", "", "", metric_type="COSINE")
        self.client.create_index(self.ddl_guide_collection, index_params)

    def create_question_sql_pair(self):
        schema = MilvusClient.create_schema(auto_id=True, primary_field="id")

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(
            field_name="sql", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="question_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )

        self.create_collection(self.qs_pair_collection, schema)

        index_params = self.client.prepare_index_params()
        index_params.add_index("question_vector", "", "", metric_type="COSINE")
        self.client.create_index(self.qs_pair_collection, index_params)

    def start(self):
        self.create_ddl_collection()
        self.create_doc_collection()
        self.create_ddl_guide_collection()
        self.create_question_sql_pair()

    def insert_into_collection(self, collection_name, data: List[Dict]):
        return self.client.insert(collection_name, data)

    def insert_ddl_statements(self, ddls: List[str]):
        data = [
            {
                "table_name": ddl.split("\n")[0],
                "name_vector": self.embedding_model.encode_documents(
                    [ddl.split("\n")[0]]
                )[0],
                "table_ddl": ddl,
            }
            for ddl in ddls
        ]

        return self.insert_into_collection(self.ddl_collection, data)

    def insert_docs(self, docs: List[str]):
        data = [
            {"doc": doc, "doc_vector": self.embedding_model.encode_documents([doc])[0]}
            for doc in docs
        ]

        return self.insert_into_collection(self.doc_collection, data)

    def insert_question_sql_pair(self, list_pairs: List[Tuple[str, str]]):
        data = [
            {
                "question": question,
                "sql": sql,
                "question_vector": self.embedding_model.encode_documents([question])[0]
            }
            for question, sql in list_pairs
        ]

        return self.insert_into_collection(self.qs_pair_collection, data)
    
    def insert_ddl_guides(self, guides: List[Tuple[str, List]]):
        data = [
            {
                "guide": guide,
                "table_names": ", ".join(tables),
                "guide_vector": self.embedding_model.encode_documents([guide])[0],
            }
            for guide, tables in guides
        ]

        return self.insert_into_collection(self.ddl_guide_collection, data)

    @staticmethod
    def _extract_query_results(results):
        if len(results) == 0:
            return ''
        
        if isinstance(results[0], tuple):
            res = ''
            for q, sql in results:
                res += f'question: {q}\n'
                res += f'sql: {sql}\n'
                res += ('-'*10 + '\n')
            res += '\n\n'

            return res
        
        return '\n'.join(results)

    def get_related_ddl_guides(self, question) -> List:
        search_vector = self.embedding_model.encode_documents([question])

        ann_search = self.client.search(
            collection_name=self.ddl_guide_collection,
            anns_field="guide_vector",
            data=search_vector,
            limit=5,
            output_fields=["guide", "table_names"],
        )

        ann_search = {
            doc["entity"]["guide"]: doc["entity"]["table_names"]
            for doc in ann_search[0]
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

        return create_statements

    def get_related_docs(self, question) -> List:
        search_vector = self.embedding_model.encode_documents([question])

        ann_search = self.client.search(
            collection_name=self.doc_collection,
            anns_field="doc_vector",
            data=search_vector,
            limit=10,
            output_fields=["doc"],
        )

        return self._extract_query_results([doc["entity"]["doc"] for doc in ann_search[0]])

    def get_related_ddls(self, question) -> List:
        search_vector = self.embedding_model.encode_documents([question])

        ann_search = self.client.search(
            collection_name=self.ddl_collection,
            anns_field="name_vector",
            data=search_vector,
            limit=1,
            output_fields=["table_ddl"],
        )

        related_ddls = [ddl["entity"]["table_ddl"] for ddl in ann_search[0]]

        return related_ddls

    def get_many_related_ddls(self, list_question: List[str]) -> List:       
        return self._extract_query_results(
            [self.get_related_ddls(question)[0] for question in list_question]
        )

    def get_related_question_sql_pair(self, question):
        search_vector = self.embedding_model.encode_documents([question])

        ann_search = self.client.search(
            collection_name=self.qs_pair_collection,
            anns_field="question_vector",
            data=search_vector,
            limit=5,
            output_fields=["sql", "question"],
        )

        related_pairs \
            = [(pair["entity"]["question"], pair["entity"]["sql"]) for pair in ann_search[0]]

        return self._extract_query_results(related_pairs)

    def get_row_by_ids(self, ids: List, collection_name):
        return self.client.query(
            collection_name=collection_name,
            ids=ids,
        )


class Rag2SQL_Model(MilvusDB_VectorStore, LLM_Model):
    def __init__(self, db_name, model_name_or_path):
        MilvusDB_VectorStore.__init__(self, db_name)
        LLM_Model.__init__(self, model_name_or_path)
        self.connection = None
        self.connection_string = ''

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

            self.connection_string \
                = f"postgresql://{user_encoded}:{password_encoded}@{host}:{port}/{dbname}"

        except Exception as e:
            print(f"The error '{e}' occurred")

    def get_ddls(self):
        db = SQLDatabase.from_uri(
            self.connection_string,
            sample_rows_in_table_info = 0,
            view_support = True,
        )

        return str(db.get_context()['table_info']).split('\n\n')

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
                print(f"The error '{e}' occurred")
                raise Exception("Error when query: ", e)
        
    @staticmethod
    def _extract_sql(sql):
        sensitive_keywords = ['DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'INSERT', 'UPDATE']
    
        for keyword in sensitive_keywords:
            if keyword in sql.upper():
                return "SELECT 'Tôi không biết' as answer;"
        
        return sql

    def generate_query(self, question):
        guides = self.get_related_ddl_guides(question)
        ddls = self.get_many_related_ddls(guides)
        docs = self.get_related_docs(question)
        question_sql_pair = self.get_related_question_sql_pair(question)

        query = self.submit_prompt(question, docs, ddls, question_sql_pair)

        return self._extract_sql(query)

    def ask(self, question):
        try:
            sql = self.generate_query(question)
        except Exception as e:
            print('Error when generate sql: ', e)
            return 'Tôi chịu'
        
        try:
            res = self.run_sql(sql)
            return res
        except Exception as e:
            print('Error when run sql: ', e)
            return 'Đang có trục trặc gì ấy.. .-.'
    
    def train(
            self, 
            ddls: List[str] = None, 
            guides: List[Tuple[str, List]] = None, 
            docs: List[str] = None,
            question_sql_pairs: List[Tuple[str, str]] = None
        ):

        self.start()

        if ddls:
            print('DDLs: ', self.insert_ddl_statements(ddls))
        if guides:
            print('Guides: ', self.insert_ddl_guides(guides))
        if docs:
            print('Docs: ', self.insert_docs(docs))
        if question_sql_pairs:
            print('Q&S Pair: ', self.insert_question_sql_pair(question_sql_pairs))        

if __name__ == "__main__":

    rag2sql = Rag2SQL_Model('milvus_demo.db', 'model_name')

    ddls = rag2sql.get_ddls()
    guide_docs = [
        ("phòng, department", ["CREATE TABLE hr_department"]),
        ("bệnh nhân, patient", ["CREATE TABLE medical_patient"]),
        ("nhân viên, staff, employee", ["CREATE TABLE hr_employee"]),
        (
            "xét nghiệm, kết quả, chỉ số, test result, test indices",
            [
                "CREATE TABLE medical_test",
                "CREATE TABLE medical_test_result",
                "CREATE TABLE medical_test_indices",
            ],
        ),
        (
            "đơn thuốc, toa thuốc, prescription",
            [
                "CREATE TABLE medical_prescription_order",
                "CREATE TABLE medical_prescription_order_line",
            ],
        ),
        ("sản phẩm, product", ["CREATE TABLE product_product"]),
    ]
    docs = ['ngày khám đầu = date_first', 'địa điểm/nơi cưới = marriage_registration_place']
    question_sql_pair = [('có bao nhiêu bệnh nhân họ Nguyễn',"SELECT first_name FROM medical_patient mp WHERE unaccent(mp.first_name) ILIKE unaccent('%nguyen%');")]

    rag2sql.train(
        ddls = ddls,
        guides = guide_docs,
        docs = docs,
        question_sql_pairs=question_sql_pair
    )

    rag2sql.ask('Có bao nhiêu bệnh nhân nam')
