import os
import time
import json
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymupdf4llm import to_markdown
import pinecone

load_dotenv()
EMBEDDING_MODEL = "dragonkue/BGE-m3-ko"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "chatbot-project"
CONFIG_PATH = "config.json"

def process_pdf(config_path):
    # config 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Pinecone 접속
    print("DB 접속 중...")
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # 현재 청크 개수 확인
    prev_count = index.describe_index_stats()['total_vector_count']
    print(f"현재 문서 개수: {prev_count}")

    docs = []

    # pdf 처리
    for file in config:
        path = file['file_path']
        
        if not os.path.exists(path):
            print(f"파일 없음: {path}") 
            continue

        print(f"처리 중: {path}")
        
        
        common_meta = file.get("common_metadata", {})

        for section in file['sections']:
            start = section['start_page']
            end = section['end_page']

            # 텍스트 추출
            pages_list = list(range(start, end + 1))
            
            if not pages_list:
                continue
            text = to_markdown(path, pages=pages_list)
            text = text.replace('\uFFFD', ' ').replace('\u0001', ' ')

            if not text.strip():
                continue

            
            ## 메타데이터 합치기
            final_meta = common_meta.copy()
            final_meta.update(section.get("metadata", {}))
            final_meta['source'] = os.path.basename(path)

            # 청크 분할
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([text])

            # 메타데이터 주입
            for chunk in chunks:
                chunk.metadata.update(final_meta)
            
            docs.extend(chunks)
        

    if not docs:
        print("추가할 내용 없음")
        return

    # 순번 부여
    print(f"총 {len(docs)}개 청크 업로드")
    for i, chunk in enumerate(docs):
        chunk.metadata['seq_num'] = prev_count + i + 1
    
    # 업로드 
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    # 100개씩 나눠서
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vectorstore.add_documents(batch)

    time.sleep(2)
    total = index.describe_index_stats()
    print(f"완료(총 문서 수: {total['total_vector_count']})")


if __name__ == "__main__":
    process_pdf(CONFIG_PATH)