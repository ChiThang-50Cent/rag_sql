from pymilvus import model
from pymilvus import MilvusClient
from utils.parse_ddl import parse_sql_file

client = MilvusClient("milvus_demo.db")


def insert_db(embedding_model, collection, docs):
    if client.has_collection(collection_name=collection):
        client.drop_collection(collection_name=collection)

    client.create_collection(
        collection_name=collection,
        dimension=384,
    )

    embedding = embedding_model.encode_documents([doc[0] for doc in docs])

    data = [
        {"id": i, "vector": embedding[i], "doc": docs[i][0], "table": docs[i][1]}
        for i in range(len(embedding))
    ]

    return client.insert(collection_name=collection, data=data)


if __name__ == "__main__":
    sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2", device="cpu"
    )

    ddl_docs = parse_sql_file("./ddl_statement/table_info.txt")
    ddl_docs = [(doc[0].split("\n")[0], doc) for doc in ddl_docs]
    print(insert_db(sentence_transformer_ef, "ddl_statements", ddl_docs))

    doc_docs = [
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
    print(insert_db(sentence_transformer_ef, "doc_statements", doc_docs))


