import os
import json
import re
import google.generativeai as genai
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document


load_dotenv()

class StreamlitRAGChatbot:
# RAG(Vector DB + Knowledge graph)기반 챗봇

    def __init__(self):
        self.INDEX_NAME = "chatbot-project"
        self.EMBEDDING_MODEL_NAME = "dragonkue/BGE-m3-ko"
        self.LATEST_YEAR = 2025
        self.MODEL_NAME = "gemini-2.5-flash"
        
        self.DEPARTMENT_TO_COLLEGE_MAP = {
            '컴퓨터공학과': '소프트웨어융합대학 공통',
            '인공지능학과': '소프트웨어융합대학 공통',
            '소프트웨어융합학과': '소프트웨어융합대학 공통',
        }

        # Google Gemini 설정
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key) # Router용
        
        self.llm = ChatGoogleGenerativeAI(    # 답변 생성용
            model=self.MODEL_NAME, 
            google_api_key=self.api_key,
            temperature=0
        )

        # Neo4j(KG) 설정
        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD"))
        self.neo4j_driver = GraphDatabase.driver(self.NEO4J_URI, auth=self.NEO4J_AUTH)

        #  Pinecone(Vector) 설정
        embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL_NAME)
        self.vectorstore = PineconeVectorStore.from_existing_index(self.INDEX_NAME, embeddings)

        #  응답 프롬프트 설정
        self.prompt = ChatPromptTemplate.from_template("""
        ### 역할 및 지시사항 ###
        아래 [학칙 및 규정] 내용을 근거로, 학생(사용자)의 질문에 명확하게 답변하세요.

        [답변 가이드라인]
        1. "문서에 따르면~"과 같은 불필요한 말은 모두 빼고, 필요한 내용만 사용자의 질문에 맞게 대답합니다.
        2. 핵심 내용을 중심으로 가독성있게 답변하세요.                                               
        3. 질문의 맥락을 파악하기 위해 필요하다면 [이전 대화] 내용을 참고하세요.                                               
        4. [학사규정]에 근거한 내용으로만 답변하세요. 없는 내용을 지어내지 마세요.
        5. 정보가 없을 때는 "죄송합니다. 현재 가지고 있는 문서에는 해당 내용이 나와있지 않습니다."라고 답변하세요.
        6. 학생의 [입학년도]와 [학과]를 고려하여 해당 학생에게 적용되는 규정을 우선적으로 설명하세요.
        7. 답변 생성이 끝난 후, 답변의 맨 마지막 줄에 실제로 참고한 규정의 '연도'를 아래 형식으로 반드시 표기하세요.
            - 형식: [[REF: 2021, 2023]] (참고한 연도가 없다면 [[REF: ]] 라고 적으세요)

        ----------
        [학사규정]
        {context}
       
        [이전 대화]
        {history}
                                                       
        [질문]
        {input}

        [학생 정보]
        - 입학년도: {admission_year}
        - 학과: {department}
        - 전공 유형: {major_type}
        -----------                                              
        """)

        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def get_departments(self):
        return list(self.DEPARTMENT_TO_COLLEGE_MAP.keys())

    # ============================================================
    # 1. 질문 유형 파악
    # ============================================================
    def analyze_intent(self, user_query, history_text):
        """
        질문을 분석하여 분류
        1. 여러 연도의 정보를 비교해야하는 질문 -> kg
        2. 그 외의 질문 -> vector
        """
        prompt = f"""
        [이전 대화]와 [질문]을 분석하여 다음 두 가지를 수행하세요.
        
        [이전 대화]
        {history_text}

        [질문]
        {user_query}

        --- 1. final_query 생성 ---
        [이전 대화]의 맥락을 반영하여 [질문]의 생략된 의미를 살려 다시 작성하세요.
        [이전 대화]가 없거나 [질문]이 이전대화와 독립적이라면 수정하지 말고 그대로 두세요.
        
        <예시>
        Case 1. 주제 연결
        - 이전: "알고리즘 선수과목이 뭐야?"
        - 현재: "자료구조는?"
        -> final_query: "자료구조의 선수과목은 무엇인가요?" (앞의 '선수과목'이라는 주제를 이어받음)

        Case 2. 대명사 해결
        - 이전: "졸업하려면 영어과목 수강해야돼?"
        - 현재: "그거 몇 과목 수강해야돼?"
        -> final_query: "영어과목 몇 과목 수강해야되나요?"

        Case 3. 독립적 질문 (수정 금지)
        - 이전: "수강신청 언제야?"
        - 현재: "장학금 신청 기간 알려줘"
        -> final_query: "장학금 신청 기간 알려줘" (문맥이 이어지지 않으므로 그대로 둠)

        
        --- 2. 검색 tool 결정 ---
        - VectorDB의 한계 
            현재 시스템의 Vector DB는 아래 두 가지 정보만 검색할 수 있습니다.
            1. 사용자의 입학년도에 해당하는 정보
            2. 가장 최신 연도(2025년)의 정보  
            따라서 위 두 연도를 제외한 다른 연도의 정보나, 여러 연도 간의 비교는 Vector DB로 처리할 수 없습니다.
        
        - 분류 기준 
            1) Knowledge graph
             - 서로 다른 연도의 졸업요건이나 교육과정을 비교하는 질문
             - Vector DB로 검색할 수 없는 특정 연도에 대한 질문
             - 교육과정/졸업요건 변경에 대한 질문
             예) "2020년도 졸업요건과 2023년도 졸업요건의 차이점이 무엇인가요?"
             예) "24 교육과정으로 변경 시 어떤 점이 유리한가요?

            2) Vector DB
                - 위 1번 조건에 해당하지 않는 모든 질문
        
                
        --- 출력 형식 (JSON) ---
        {{
            "final_query": "문맥이 반영된 완성된 질문",
            "tool": "KG" 또는 "Vector"
        }}
        """
        
        model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            system_instruction= prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        try:
            response = model.generate_content(user_query)
            return json.loads(response.text)
        except:
            return {"tool": "Vector"} 

    # ============================================================
    #  2. KG 데이터 검색(비교 질문에 사용):
    #       사용자가 선택한 학과와 전공유형에 해당하는 모든 연도의 졸업요건 데이터를 Neo4j에서 가져옴
    # ============================================================
    def get_kg_data(self, department, major_type):
    

        cypher_query = """
        MATCH (req:Requirement {department: $dept, major_type: $type})
        MATCH (req)-[r:INCLUDES]->(sub:Subject)
        RETURN 
            req.year AS year,
            properties(req) AS req_props,
            properties(r) AS rel_props,
            properties(sub) AS sub_props
        ORDER BY req.year ASC
        """
        
        params = {"dept": department, "type": major_type}
        
        context_text = ""
        with self.neo4j_driver.session() as session:
            records = session.run(cypher_query, params)
            
            data_list = []
            for record in records:
                item = {
                    "연도": record['year'],
                    "졸업요건_요약": record['req_props'],
                    "과목정보": {
                        "과목명": record['sub_props'].get('name'),
                        "학수번호": record['sub_props'].get('id'),
                        "이수구분": record['rel_props'].get('classification'),
                        "상세구분": record['rel_props'].get('sub_classification')
                    }
                }
                data_list.append(item)
            
            json_str = json.dumps(data_list, ensure_ascii=False, indent=2)
            
            return [Document(page_content=json_str, metadata={"source": "소프트웨어융합대학 교육과정 문서"})]
    
    # ============================================================
    #  2-1. KG 데이터 검색(일반 질문에 사용):
    #       사용자 정보에 해당하는 졸업요건 노드만 가져옴
    # ============================================================
    def get_user_subgraph(self, year, dept, major_type):
        query = """
        MATCH (req:Requirement {year: $year, department: $dept, major_type: $type})
        RETURN properties(req) AS info
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(query, year=int(year), dept=dept, type=major_type)
            record = result.single() 
            
            if record:
                return str(record["info"])
            return ""
        

    # ============================================================
    # 3. VectorDB 데이터 검색(일반 질문에 사용):
    #       사용자가 선택한 학과와 연도를 기준으로 검색
    # ============================================================
    def get_vector_context(self, admission_year, department, query):
        college = self.DEPARTMENT_TO_COLLEGE_MAP.get(department)
        
        # 1차 검색: 사용자가 선택학 연도의 문서 검색
        filter_primary = {  
            "$and": [
                {"year": {"$eq": admission_year}},
                {"$or": [{"department": {"$eq": department}}, {"department": {"$eq": college}}]}
            ]
        }

        # 2차 검색: 가장 최근 연도 문서 검색(개편된 정보 반영하기 위함)
        filter_secondary = {
            "$and": [
                {"year": {"$eq": self.LATEST_YEAR}},
                {"$or": [{"department": {"$eq": department}}, {"department": {"$eq": college}}]}
            ]
        }
        
        retriever_p = self.vectorstore.as_retriever(search_kwargs={'k': 8, 'filter': filter_primary})
        retriever_s = self.vectorstore.as_retriever(search_kwargs={'k': 8, 'filter': filter_secondary})
        
        docs = retriever_p.invoke(query) + retriever_s.invoke(query)
        
        unique_docs = { (doc.metadata['source'], doc.metadata.get('seq_num', 0)): doc for doc in docs }
        return list(unique_docs.values())
    
    # ============================================================
    # 4. 남은 학점 계산기(자가 졸업진단 기능)
    # ============================================================
    def check_graduation_status(self, year, dept, major_type, taken_subjects_list):
        # 1. 입력값 Set 변환
        taken_set = set(taken_subjects_list)

        # 2. DB 쿼리 
        query = """
        MATCH (req:Requirement {year: $year, department: $dept, major_type: $type})
        MATCH (req)-[r:INCLUDES]->(subject:Subject)
        WHERE r.classification IN ['전공필수', '전공기초', '전공선택'] 
        OPTIONAL MATCH (subject)-[s:SUBSTITUTES]->(alternative:Subject)
        RETURN 
            properties(req) AS req_props, 
            r.classification AS classification,        
            r.sub_classification AS sub_classification, 
            subject.name AS subject_name, 
            subject.aliases AS subject_aliases, 
            subject.credits AS subject_credits,
            alternative.name AS alternative_name, 
            alternative.aliases AS alternative_aliases, 
            s.note AS note
        """
        
        with self.neo4j_driver.session() as session:
            data = [dict(r) for r in session.run(query, year=int(year), dept=dept, type=major_type)]

        # 3. 초기화
        req_info = data[0]['req_props'] if data else {}
        missing = {}
        earned = {} 
        processed = set()

        for r in data:
            cls = r['classification']          
            sub_cls = r['sub_classification']  
            subj_name = r['subject_name']
            credits = r['subject_credits'] or 0
            
            # 이수 여부 체크
            candidates = [subj_name] + (r['subject_aliases'] or []) + \
                         ([r['alternative_name']] if r['alternative_name'] else []) + \
                         (r['alternative_aliases'] or [])
            is_taken = any(c in taken_set for c in candidates if c)

            # 학점 계산
            if is_taken:    # 남은 학점 계산
                if subj_name not in processed:
                    earned[cls] = earned.get(cls, 0) + credits
                    if sub_cls == '산학필수':
                        earned['산학필수'] = earned.get('산학필수', 0) + credits
                    processed.add(subj_name)
            else:   # 남은 과목 계산(전기, 전필)
                # 중복 제거하여 대체과목도 표시
                if cls in ["전공필수", "전공기초"]:
                    missing_list = missing.setdefault(cls, [])
                    existing = next((x for x in missing_list if x["name"] == subj_name), None)
                    
                    current_alt = r['alternative_name']

                    if existing:
                        if current_alt:
                            if existing["alternatives"] == "없음":
                                existing["alternatives"] = current_alt
                            elif current_alt not in existing["alternatives"]:
                                existing["alternatives"] += f", {current_alt}"
                    else:
                        entry = {
                            "name": subj_name, 
                            "credits": credits, 
                            "alternatives": current_alt or "없음", 
                            "note": r['note'] or ""
                        }
                        missing_list.append(entry)

        # 최종 현황 집계
        status = {}
        # DB 속성 매핑
        mapping = {'credits_major_required': '전공필수', 'credits_major_elective': '전공선택', 
                   'credits_major_basic': '전공기초', 'credits_industry_required': '산학필수'}

        for db_k, kor in mapping.items():
            req_score = req_info.get(db_k, 0) or 0
            cur_score = earned.get(kor, 0)
            status[kor] = {'required': req_score, 'earned': cur_score, 'remaining': max(0, req_score - cur_score)}

        return req_info, missing, status

    # ============================================================
    #  4. 메인 Chat 함수
    # ============================================================
    def chat(self, admission_year: int, department: str, query: str, history=None, major_type="단일전공"):
        
        # 1. 히스토리 포맷팅
        history_text = ""
        if history:
            recent = history[-6:]   # 앞의 3개 대화까지 기억
            formatted = [f"{'사용자' if m.get('role')=='user' else '챗봇'}: {m.get('content','')}" for m in recent]
            history_text = "\n".join(formatted)
        else:
            history_text = "이전 대화 없음."

        # 2. 의도 파악
        intent_result = self.analyze_intent(query, history_text) 
        tool = intent_result.get("tool", "Vector")
        final_query = intent_result.get("final_query", query)
        print(f"Original: {query} -> Refined: {final_query}")   #디버깅용
        
        # 3. 데이터 검색 (KG 또는 Vector)
        docs = []
        source_data = ""
        
        if tool == "KG":
            docs = self.get_kg_data(department, major_type)
            source_data = "소프트웨어융합대학 교육과정 PDF"
        else:
            docs = self.get_vector_context(admission_year, department, final_query) # 검색시에는 다시 생성된 쿼리로 
             # ===== 디버깅 =====
            print(f"검색된 문서 수: {len(docs)}")
            if docs:
                for i, doc in enumerate(docs[:3]):
                    print(f"\n--- Doc {i+1} ---")
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"Year: {doc.metadata.get('year', 'Unknown')}")
                    print(f"Dept: {doc.metadata.get('department', 'Unknown')}")
                    print(f"Content (first 150 chars): {doc.page_content[:150]}")
            else:
                print("검색된 문서 0개")
    # ===== 끝 =====
            kg_data = self.get_user_subgraph(admission_year, department, "졸업요건")

            # 쿼리에 kg 데이터 같이 포함시킴
            if kg_data:
                query = f"[중요 참고사항(사용자 졸업요건 정보)]\n{kg_data}\n\n[질문]\n{query}"
            
            chunk_nums = sorted([int(d.metadata.get('seq_num', 0)) for d in docs if d.metadata.get('seq_num') is not None])
            source_data = ", ".join(map(str, chunk_nums)) if chunk_nums else "없음"

        # 4. 답변 생성
        if not docs:
            return "관련된 정보를 찾을 수 없었습니다.", "[]"
        
        numbered_docs = []
        for i, doc in enumerate(docs):
            
            doc_year = doc.metadata.get("year")
        
            # 각 청크 앞에 번호와 연도를 적어서 llm에 전달
            new_content = f"[{i+1}] ({doc_year}년 규정) {doc.page_content}"
            
            new_doc = Document(
                page_content= new_content,
                metadata= doc.metadata
            )
            numbered_docs.append(new_doc)
            
        response = self.document_chain.invoke({
            "input": query, # 답변 생성시에는 원래 쿼리로
            "context": numbered_docs,  
            "admission_year": admission_year,
            "department": department,
            "major_type": major_type,
            "history": history_text
        }) 
        
        # 5. 답변 출처 필터링
        selected_docs = []
        
        if "[[REF:" in response:
            # 답변만 남김
            text, ref_str = response.split("[[REF:", 1)
            response = text.strip() 
            
            # llm로부터 받은 출처번호 유효성 확인
            indices = [int(n)-1 for n in re.findall(r'\d+', ref_str)]
            selected_docs = [docs[i] for i in indices if 0 <= i < len(docs)]
        
        if not selected_docs:
            selected_docs = docs[:3]

        # 6. UI 출처 표시
        source_data = []
        seen = set()
        page = None
        
        for doc in selected_docs:
            source = doc.metadata.get("source", "알 수 없음")
            
            # 웹사이트
            if source.startswith("http") or source.startswith("www"):
                name = "홈페이지" 
                url = source       
                page = None            
            
            # 파일
            else:
                name = source.split("/")[-1] 
                url = None  
                page = None 
            
            # 중복 제거하고 리스트에 추가
            if (name, page, url) not in seen:
                source_data.append({"name": name, "page": page, "url": url})
                seen.add((name, page, url))

        return response, source_data
