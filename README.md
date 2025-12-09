# RAG 기반 학사정보 챗봇

> LLM(Google Gemini)과 Knowledge Graph(Neo4j), Vector DB(Pinecone)를 결합한 학사정보 질의응답 시스템

---


## 1. 프로젝트 개요
* **과목명:** 졸업프로젝트
* **학기**: 2025-2학기
* **이름**: 김은결
* **학번**: 2020103145

  
### 프로젝트 배경
* 매년 변경되는 복잡한 졸업 요건과 흩어져 있는 학사 정보로 인한 학생들의 불편함 해결

### 데이터 범위
- **연도**: 2020~2025년
- **학과**: 경희대학교 소프트웨어융합대학 소속 학과(컴퓨터 공학과, 인공지능학과, 소프트웨어융합학과)


---

## 2. 핵심 기능 

### 🤖 1. 개인 맞춤형 학사 규정 Q&A
* **Hybrid Search:** 질문 의도에 따라 **Vector Search**와 **Graph Search**를 중 적합한 검색방식을 자동 분류하여 답변
* **Personalized Filtering:** 입학년도, 학과, 전공 유형(단일/다/부전공) 등 사용자 정보를 기반으로 **사용자에게 유효한 정보만 필터링**
* **출처 표시:** 답변시 근거 문서 or url 표시
* **multi-turn 대화:** 이전 대화기록을 반영한 **질문 재작성**을 통해 연속 대화 지원

### 🎓 2. 졸업 요건 자가진단 
* 수강 과목 입력 시 Neo4j 그래프에서 필요한 서브그래프를 가져와 계산
* **이수 현황 분석:** 영역별(전공필수/선택 등) 이수 학점 계산
* **미이수 과목 도출:** 졸업을 위해 필수적으로 수강해야 하는 잔여 과목 안내
* **대체 과목 추론:** 과목과 연결된 **'대체 (Substitutes)'** 관계 혹은 과목의 **'별칭(aliases)'** 속성을 파악하여 구과목/대체인정과목도 이수 처리

---

## 3. 시스템 아키텍처 및 기술 스택 

### 🛠 Tech Stack
| Category | Tech |
|:---:|:---|
| **Language** | Python |
| **Frontend** | Streamlit |
| **LLM** | Google Gemini 2.5 Flash |
| **Framework** | LangChain |
| **Vector DB** | Pinecone |
| **Knowled Graph** | Neo4j AuraDB |
| **Embedding** | HuggingFace (BGE-m3-ko) |

---

## 4. 프로젝트 구조 
```bash
├── app.py                  # Streamlit 프론트엔드 실행 파일
├── backend.py              # RAG 챗봇 로직 (질문 분류, 검색, 응답 생성)
├── data/                   # 교육과정 PDF (사용한 원본 데이터)
├── vector_db/              # Vector DB 구축 관련
│   ├── create_db.py            # PDF 기반 DB 구축
│   ├── update_db_from_web.py   # 웹페이지 기반 DB 업데이트
│   └── config.json             # PDF 페이지 설정 파일 (메타데이터 정의)
├── kg/                     # Knowledge Graph 구축 관련
│   ├── extract_tables.py       # PDF 내 표 추출
│   ├── extract_tables_includes.py  # PDF 내 표 추출 (Includes 관계)
│   ├── create_subject.py       # 과목 노드 생성 (LLM 활용)
│   ├── create_requirement.py   # 졸업요건 노드 생성 (LLM 활용)
│   ├── create_includes.py      # 포함 관계 생성 (LLM 활용)
│   ├── create_substitutes.py   # 대체 과목 관계 생성 (LLM 활용)
│   ├── upload_neo4j.py         # 노드, 관계 Neo4j DB에 업로드
│   ├── update_neo4j.py         # 대체 관계 Neo4j DB에 추가 업데이트
│   ├── manifest/               # 표 추출을 위한 페이지 설정 파일들
│   └── output/                 # ETL 과정의 중간 산출물 (JSON)
├── requirements.txt
└── README.md
```

---

## 5. 실행 방법

**⚠️주의**: 실행하기 위해서는 사전 구축된 Pinecone 및 Neo4j에 대한 접근 권한 필요


```bash
# 패키지 설치
pip install -r requirements.txt

# 환경 변수 설정 (.env)
GOOGLE_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_ENV=your_env 
NEO4J_URI=your_uri
NEO4J_PASSWORD=your_password

# 실행
streamlit run app.py
```

## 6. 데이터베이스 구축 과정 (DB Setup)

### 6.1: Vector DB (Pinecone) 구축

```bash
# step1. 설정 파일 준비 (config.json): pdf 범위 지정 및 메타데이터 정의 

# step2. 텍스트 추출 -> 분할 -> 임베딩 -> DB 저장 (pdf문서용)
python vector_db/create_db.py

# step3. 웹 페이지 정보 업데이트
python vector_db/update_db_from_web.py
```

### 6.2: Knowledge Graph (Neo4j) 구축

```bash
# step1. 설정 파일 준비(manifest/): pdf 범위 지정 및 메타데이터 정의

# step2. 그래프 구축에 필요한 표 추출 (includes 만 별도)
python kg/extract_tables.py 
python kg/extract_tables_includes.py

# step3. Subject, Requirement 노드 생성
python kg/create_subject.py
python kg/create_requirement.py

# step4. INCLUES 관계 생성
python kg/create_includes.py

# step5. Neo4j에 저장
python kg/upload_neo4j.py

# step6. SUBSTITUES 관계 생성 및 저장
python kg/create_substitutes.py
python kg/update_neo4j.py
```


