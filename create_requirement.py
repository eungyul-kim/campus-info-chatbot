import google.generativeai as genai
import json
import os
from dotenv import load_dotenv 

# --- 설정 ---
load_dotenv() 

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    exit("API 키가 없습니다.")
genai.configure(api_key=api_key)

INPUT = "output/requirement_tables.json" 
OUTPUT = "output/requirement_nodes.json" 

def build_prompt(chunk):
    
    metadata = json.dumps(chunk['metadata'], ensure_ascii=False, indent=2)
    table_string = chunk['table_data_as_string'] 

    prompt = f"""
    주어진 표의 내용을 기반으로 졸업요건(Requirement) 노드를 JSON으로 만들어주세요.

    [참고 정보]:
    {metadata}
    
    [입력 데이터 (Python 리스트 형태의 테이블 텍스트)]:
    {table_string}

    [요청 사항]:
    입력 데이터의 졸업 요건 표를 읽고, 아래 형식에 맞는 `Requirement` 노드를 생성하세요.
    
    --- 기본 처리 규칙 ---
    - 입력 데이터에서 졸업 요건과 관련 없는 표는 무시하세요.
    - 만약 부전공에 여러 과정이 있을 경우 첫번째 과정만 취급하고 나머지는 무시하세요.
    
    --- 전공선택 학점(credits_major_elective) 계산 규칙 ---
    - 산학필수는 전공선택의 하위분야입니다. 표에서 '전공선택' 항목 하위에 '산학필수'와 '전공선택이 나뉘어 있다면, 
      `credits_major_elective` 값은 이들을 모두 합친 값이어야 합니다.
    - 예시: 표에 `전공선택` 하위로 `산학필수(12)`와 `전공선택(15)`가 있다면,
         -> `credits_major_elective`: 27(12 + 15)로 입력하세요.

    --- 속성 목록((총 8개)) ---
    - `id`: "연도_학과명_유형" (예: "2025_컴퓨터공학과_단일전공")
    - `type`: "Requirement" (고정값)
    - `year`: 참고 정보에서 확인 (예: 2025)
    - `department`: 참고 정보에서 확인 (예: "컴퓨터공학과")
    - `major_type`: 무조건 다음 예시의 3개 값 중 하나(예: "단일전공", "다전공", "부전공")
    - `total_credits`: 총 이수 학점 (예: 130)
    - `credits_major_basic`: 전공기초 학점 (예: 12)
    - `credits_major_required`: 전공필수 학점 (예: 42)
    - `credits_major_elective`: 전공선택 학점 (예: 27)
    - `credits_industry_required`: 산학필수 학점 (예: 9), 정보가 없으면 0


    반드시 아래 형식을 따르세요:

    {{
      "nodes": [
        {{
          "id": "2025_컴퓨터공학과_단일전공",
          "type": "Requirement",
          "year": 2025,
          "department": "컴퓨터공학과",
          "major_type": "단일전공",
          "total_credits": 130,
          "credits_major_basic": 12,
          "credits_major_required": 42,
          "credits_major_elective": 27,
          "credits_industry_required": 9
        }}
        // (만약 다전공, 부전공 정보도 있으면 노드를 추가로 생성)
      ],
      "relationships": [] 
    }}
    ---

    위 규칙대로 모든 '유형'(단일, 다, 부전공)에 대한 표를 처리하여 Requirement 노드를 생성해 주세요.
    """
    return prompt


def create_requirement_nodes(chunks):
    # 텍스트 청크 리스트를 Requirement 노드로 변환

    generation_config = {"response_mime_type": "application/json"}
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

    all_nodes = []
    all_relationships = []  
    ids = set()   # 노드 중복 방지

    total = len(chunks)

    for i, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        dept = metadata.get("department")
        year = metadata.get("year")

        print(f"\n  ({i+1}/{total}) {dept} {year}년 처리 중...")

        prompt = build_prompt(chunk)

        try:
            response = model.generate_content(prompt)
            data = json.loads(response.text)
            new_nodes = data.get("nodes", [])
        except Exception as e:
            print(f"  [오류] {e}")
            continue

        # 중복 id 제거하면서 노드 추가
        for node in new_nodes:
            id = node.get("id")
            if id and id not in ids:
                all_nodes.append(node)
                ids.add(id)

    # 다전공/부전공은 총 이수학점을 null로
    for node in all_nodes:
        major_type = node.get("major_type")
        if major_type in ["다전공", "부전공"]:
            if node.get("total_credits") is not None:
                node["total_credits"] = None

    results = {
        "nodes": all_nodes,
        "relationships": all_relationships,  
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{len(all_nodes)}개 노드 생성됨")


# --- 메인 실행 ---
if __name__ == "__main__":
    try:
        with open(INPUT, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        exit(f"파일 없음: {INPUT}")

    create_requirement_nodes(chunks)
