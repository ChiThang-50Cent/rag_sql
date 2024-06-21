import time
from pymilvus import model
from pymilvus import MilvusClient
from pymilvus.model.reranker import CrossEncoderRerankFunction

client = MilvusClient("milvus_demo.db")

def get_related_doc(rerank_model, embedding_model, query):
    query_vector = embedding_model.encode_queries([query])
    
    ann_res = client.search(
        collection_name="doc_statements",
        anns_field="vector", 
        data=query_vector,
        limit=5,
        output_fields=["doc", "table"],
    )

    ann_res = {doc['entity']['doc']: doc['entity']['table'] for doc in ann_res[0]}

    reranked_results = rerank_model(
        query=query,
        documents=list(ann_res.keys()),
        top_k=2
    )

    create_statements = [ann_res[key.text] for key in reranked_results]
    create_statements = [statement for statements in create_statements for statement in statements]

    return create_statements

def get_related_ddl(embedding_model, query):
    query_vector = embedding_model.encode_queries([query])
    
    ann_res = client.search(
        collection_name="ddl_statements",
        anns_field="vector", 
        data=query_vector,
        limit=1,
        output_fields=["doc", "table"],
    )

    return ann_res[0][0]['entity']['table']

def main(embedding_model, ce_rf):
    queries = [
        'có bao nhiêu bệnh nhân nam',
        'có bao nhiêu bệnh nhân có nhiều hơn 2 xét nghiệm',
        'có bao nhiêu bệnh nhân không có toa thuốc nào',
        'có bao nhiêu bệnh nhân sinh tháng 6',
        'có bao nhiêu bệnh nhân bắt đầu khám năm 2024',
        'có bao nhiêu bệnh nhân họ Nguyễn'
    ]

    with open('output.txt', 'w') as file:
        for query in queries:
            related_doc = get_related_doc(ce_rf, embedding_model, query)
            file.write(f'----- {query} -------\n')

            for doc in related_doc:
                file.write(doc + '\n')
                file.write(''.join(get_related_ddl(embedding_model, doc)) + '\n')
        
            file.write('-' * 20 + '\n\n')
            

if __name__ == "__main__":
    start_time = time.time()

    ce_rf = CrossEncoderRerankFunction(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )

    sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2", device="cpu"
    )

    main(sentence_transformer_ef, ce_rf)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
