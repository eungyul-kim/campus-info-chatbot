import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


NEO4J_URI = os.getenv("NEO4J_URI") 
NEO4J_USER = "neo4j"                 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")   


SUBJECT_NODES = "output/subject_nodes.json"        
REQ_NODES = "output/requirement_nodes.json" 
INCLUDES_RELS = "output/includes_relationships.json" 

class Neo4jUploader:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity() 
            print("[연결 완료]")
        except Exception as e:
            print(f"[오류] Neo4j 연결 실패: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_database(self):
        # 기존 모든 노드와 관계를 삭제
        if not self.driver: return
        print("\n[Neo4j] 기존 데이터베이스를 초기화...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("    -> 초기화 완료.")

    def upload_nodes(self, nodes, label=""):
        # 노드 업로드
        if not self.driver or not nodes: 
            return

        print(f"\n[Neo4j] {label} 노드 {len(nodes)}개를 업로드...")
        
        query = """
        UNWIND $nodes_batch AS node_props
        CALL apoc.create.node([node_props.type], {id: node_props.id})
        YIELD node
        SET node += apoc.map.removeKey(node_props, 'type')
        """
        
        with self.driver.session() as session:
            try:
                for i in range(0, len(nodes), 500):
                    batch = nodes[i:i+500]
                    session.run(query, nodes_batch=batch)
            except Exception as e:
                print(f"  [오류] {label} Cypher 오류: {e}")
                
        print(f"    -> {label} 노드 {len(nodes)}개 업로드 완료")

    def upload_relationships(self, relationships, label=""):
        # 관계 업로드 
        if not self.driver or not relationships: 
            return

        print(f"\n[Neo4j] {label} 관계 {len(relationships)}개 업로드...")
        
        query = """
        UNWIND $rels_batch AS rel_props
        MATCH (a {id: rel_props.source_id})
        MATCH (b {id: rel_props.target_id})
        CALL apoc.create.relationship(a, rel_props.type, apoc.map.removeKeys(rel_props, ['source_id', 'target_id', 'type', 'target_name_raw']), b)
        YIELD rel
        RETURN count(rel)
        """

        with self.driver.session() as session:
            try:
                for i in range(0, len(relationships), 500):
                    batch = relationships[i:i+500]
                    session.run(query, rels_batch=batch)
            except Exception as e:
                print(f"  [오류] {label} 관계 업로드 중 Cypher 오류: {e}")
                
        print(f"    -> {label} 관계 {len(relationships)}개 업로드 완료.")

# --- 메인 실행 ---
if __name__ == "__main__":
    
    all_nodes = []
    all_relationships = []

    # 모든 JSON 파일 읽어서 합침
    try:
        # 1. 과목 노드 로드
        with open(SUBJECT_NODES, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_nodes.extend(data.get("nodes", []))
        
        # 2. 졸업요건 노드 로드
        with open(REQ_NODES, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_nodes.extend(data.get("nodes", []))
            
        # 3. INCLUDES 관계 로드
        with open(INCLUDES_RELS, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_relationships.extend(data.get("relationships", []))

        print(f"  -> 총 {len(all_nodes)}개의 노드, {len(all_relationships)}개의 관계가 준비")
        
        if not all_nodes or not all_relationships:
            print("[오류] 노드나 관계가 없습니다.")
            exit()

    except FileNotFoundError as e:
        print(f"[오류] JSON 파일X {e.filename}")
        exit()
    
    # Neo4j 연결
    uploader = Neo4jUploader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    if uploader.driver:
        # 기존 DB를 모두 지움
        uploader.clear_database() 
        
        # 노드, 관계 업로드
        uploader.upload_nodes(all_nodes, "Subject + Requirement")
        uploader.upload_relationships(all_relationships, "INCLUDES")
        uploader.close()
        
        print("\n[완료]")