import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone

load_dotenv()
EMBEDDING_MODEL = "dragonkue/BGE-m3-ko"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "chatbot-project"

# 처리할 URL 및 메타데이터 정의
URLS_TO_PROCESS = [
    {
        "url": "https://ce.khu.ac.kr/ce/user/contents/view.do?menuNo=1600056",
        "metadata": {
            "college": "소프트웨어융합대학",
            "department": "컴퓨터공학과",
            "year": 2025
        }
    },
    {
        "url": "https://ce.khu.ac.kr/ce/user/contents/view.do?menuNo=1600015",
        "metadata": {
            "college": "소프트웨어융합대학",
            "department": "소프트웨어융합대학 공통",
            "year": 2025
        }
    },
    {
        
        "url": "https://ce.khu.ac.kr/ce/user/bbs/BMSR00040/list.do?menuNo=1600123",
        "metadata": {
            "college": "소프트웨어융합대학",
            "department": "소프트웨어융합대학 공통",
            "year": 2025
        }
    }
]

def update_db_from_web():
    # Pinecone 접속
    print("DB 접속 중...")
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"인덱스 없음: {INDEX_NAME}")
        return
        
    index = pc.Index(INDEX_NAME)

    # 현재 청크 개수 확인
    prev_count = index.describe_index_stats()['total_vector_count']
    print(f"현재 문서 개수: {prev_count}")

    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # URL 처리
    for item in URLS_TO_PROCESS:
        url = item["url"]
        meta = item["metadata"]
        meta["source"] = url 

        print(f"처리 중: {url}")
        
        try:
            loader = WebBaseLoader([url])
            web_docs = loader.load()
            
            chunks = text_splitter.split_documents(web_docs)
            
            # 메타데이터 주입
            for chunk in chunks:
                chunk.metadata.update(meta)
            
            docs.extend(chunks)
            
        except Exception as e:
            print(f"오류 발생 ({url}): {e}")
            continue

    if not docs:
        print("추가할 내용 없음")
        return

    # 순번 부여
    print(f"총 {len(docs)}개 청크 업로드")
    for i, chunk in enumerate(docs):
        chunk.metadata['seq_num'] = prev_count + i + 1

    # 업로드
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    ).add_documents(docs)
    
    # 결과 확인
    time.sleep(5)
    total = index.describe_index_stats()
    print(f"완료 (총 문서 수: {total['total_vector_count']})")

if __name__ == "__main__":
    update_db_from_web()