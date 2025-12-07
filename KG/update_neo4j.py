import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


NEO4J_URI = os.getenv("NEO4J_URI") 
NEO4J_USER = "neo4j"                 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

FILE_NEW_NODES = "step2_new_subjects.json"    # 대체과목 생성하면서 새로 발견된 구 과목들
FILE_SUB_RELS = "step2_substitutes.json"      # 대체 관계들

class Neo4jAppender:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity() 
        print("[Neo4j] 추가 업로드를 위해 연결되었습니다.")

    def close(self):
        if self.driver:
            self.driver.close()

    # 노드 추가 (MERGE 사용: 있으면 무시, 없으면 생성)
    def append_nodes(self, nodes):
        if not nodes: 
            print("  [정보] 추가할 노드가 없습니다.")
            return

        print(f"\n[Neo4j] 새 과목 노드 {len(nodes)}개를 추가(MERGE)합니다...")
        
        query = """
        UNWIND $batch AS row
        MERGE (n:Subject {id: row.id})
        ON CREATE SET
            n.name = row.name,
            n.credits = row.credits,
            n.credits_note = row.credits_note,
            n.type = row.type
        ON MATCH SET
            n.credits = row.credits  // (선택) 이미 있어도 학점 정보는 최신으로 업데이트
        """
        
        with self.driver.session() as session:
            try:
                for i in range(0, len(nodes), 500):
                    batch = nodes[i:i+500]
                    session.run(query, batch=batch)
                print(f"    -> 노드 처리 완료")
            except Exception as e:
                print(f"    [오류] 노드 추가 중 실패: {e}")

    # 대체 관계 추가
    def append_relationships(self, rels):
        if not rels:
            print("추가할 관계가 없음")
            return

        print(f"\n대체 관계 {len(rels)}개 추가")
        
        # Source와 Target을 찾아서 연결
        query = """
        UNWIND $batch AS row
        MATCH (s:Subject {id: row.source_id})
        MATCH (t:Subject {id: row.target_id})
        MERGE (s)-[r:SUBSTITUTES]->(t)
        ON CREATE SET
            r.department = row.department,
            r.year = row.year,
            r.note = row.note
        """

        with self.driver.session() as session:
            try:
                for i in range(0, len(rels), 500):
                    batch = rels[i:i+500]
                    session.run(query, batch=batch)
                print(f"    -> 관계 처리 완료.")
            except Exception as e:
                print(f"    [오류] 관계 추가 중 실패: {e}")

if __name__ == "__main__":
    
    # 1. 파일 로드
    new_nodes = []
    sub_rels = []

    try:
        if os.path.exists(FILE_NEW_NODES):
            with open(FILE_NEW_NODES, 'r', encoding='utf-8') as f:
                data = json.load(f)
                new_nodes = data.get("nodes", []) if isinstance(data, dict) else data

        if os.path.exists(FILE_SUB_RELS):
            with open(FILE_SUB_RELS, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sub_rels = data.get("relationships", []) if isinstance(data, dict) else data
                
    except Exception as e:
        print(f"[오류] 파일 로드 실패: {e}")
        exit()

    print(f"노드 {len(new_nodes)}개, 관계 {len(sub_rels)}개 로드됨")

    # 2. 업로드 실행
    appender = Neo4jAppender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # 노드 -> 관계 순서로 
    appender.append_nodes(new_nodes)      
    appender.append_relationships(sub_rels)
    
    appender.close()
    print("\n[완료]")