import google.generativeai as genai
import json
import os
from dotenv import load_dotenv 

load_dotenv() 

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    exit("API 키가 없습니다.")
genai.configure(api_key=api_key)

INPUT = "output/subject_tables.json" 
OUTPUT = "output/subject_nodes.json" 

def build_prompt(chunk):
    
    metadata = json.dumps(chunk['metadata'], ensure_ascii=False, indent=2)
    table_string = chunk['table_data_as_string'] 

    prompt = f"""
    주어진 표의 내용을 기반으로 과목(Subject) 노드를 JSON으로 만들어주세요.

    [참고 정보]:
    {metadata}
    
    [입력 데이터 (Python 리스트 형태의 테이블 텍스트)]:
    {table_string}

    [요청 사항]:
    아래 기준으로 학수번호(id), 과목명(name), 학점(credits)을 추출해
    Subject 노드를 리스트(JSON) 형태로 정리하세요.

    --- 과목명(name) 처리 규칙 ---
    1. 과목명 뒤에 있는 `(인공지능)`, `(SWCON)`, `(CSE)` 등 모든 괄호와 그 안의 내용을 제거하고 순수한 과목명만 남기세요.
        - 예: "자료구조(SWCON)" -> "자료구조"
    2. 숫자는 과목명 일부라면 남겨도 됩니다.
        - 예: "독립심화학습(인공지능)1" -> "독립심화학습1" 
    
    --- 학수번호(id) 처리 규칙 ---
    1. 학수번호가 'CSE101'처럼 정상적으로 존재하면, 그대로 사용하세요.
    2. 학수번호가 없으면 과목명을 id로 사용하세요.
    3. 둘 다 없으면 제외하세요.

    --- 학점(credits) 처리 규칙 ---
    1. 학점이 숫자 한 개일 경우(예: "3") → credits = 3, credits_note = null
    2. 학점이 복잡한 문자열인 경우(예: "3/12") → credits = 0, credits_note = 원본 문자열
    3. 비어있으면 → credits = 0, credits_note = null
    
    반드시 아래 형식을 따르세요:
    
    {{
      "nodes": [
        {{
          "id": "CSE101",        // (학수번호가 있으므로 id로 사용)
          "type": "Subject",
          "name": "자료구조",
          "credits": 3,
          "credits_note": null
        }},
        {{
          "id": "창업현장실습",     // (학수번호가 없으므로 'name'을 id로 사용)
          "type": "Subject",
          "name": "창업현장실습",
          "credits": 0,
          "credits_note": "3/12"
        }}
      ],
      "relationships": []
    }}
    ---
    
    위 규칙대로 전체 표를 처리하여 Subject 노드 리스트를 생성해 주세요.
    """
    
    return prompt


def create_subject_nodes(chunks):
    # 텍스트 청크 리스트를 Subject 노드로 변환

    generation_config = {"response_mime_type": "application/json"}
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

    nodes_map = {} #id 기준으로 노드 + aliases 관리

    for i, chunk in enumerate(chunks):
        metadata = chunk['metadata']
        year = metadata.get('year')
        print(f"\n  ({i+1}/{len(chunks)}) {metadata.get('department')} {year}년 처리 중...")
        
        prompt = build_prompt(chunk)
            
        try:
            response = model.generate_content(prompt)
            new_nodes = json.loads(response.text).get('nodes', [])
            
            for node in new_nodes:
                id = node.get('id')
                name = node.get('name')

                if not id: continue

                # 줄바꿈 -> 공백으로 변경, 주석 제거, 양쪽 공백 제거
                id = str(id).split('※')[0].split('*')[0].replace('\n', '').strip()
                name = str(name).split('※')[0].split('*')[0].replace('\n', ' ').strip()
                name = " ".join(name.split())
                
                # 정제된 id, name으로 노드 정보 갱신
                node['id'] = id
                node['name'] = name
                
                # 학점 처리
                try: node['credits'] = int(node['credits'])
                except: node['credits'] = 0

                # 학수번호는 같지만 과목명이 바뀐 경우 처리
                if id not in nodes_map:  # 기존에 없는 과목
                    node['aliases'] = []
                    nodes_map[id] = node 
                else:   # 이미 있는 과목
                    existing = nodes_map[id]
                    
                    # 비교를 위해 공백 제거
                    new = name.replace(" ", "")
                    main = existing['name'].replace(" ", "")
                    aliases = [a.replace(" ", "") for a in existing['aliases']]
                    
                    # 기존의 과목명과 다르고, 기존 별칭 리스트에도 없는 경우만 별칭 추가
                    if new != main:
                        if new not in aliases:
                            existing['aliases'].append(name)
                            print(f"  alias 추가: {id} ({existing['name']} ← {name})")

        except Exception as e:
            print(f"  [오류] {e}")
            continue 

    all_nodes = list(nodes_map.values())

    # 현장실습 예외 처리(credits_note 추가)
    SPECIAL_CREDITS_NOTE = "창업현장실습,단기현장실습/장기현장실습은 각 활동별로 3학점, 6학점, 9학점, 12학점을 산학필수학점으로 이수함"
    TARGET_SUBJECT_IDS = ["창업현장실습", "단기현장실습", "장기현장실습", "현장실습"]

    for node in all_nodes:
        if node.get("id") in TARGET_SUBJECT_IDS:
            node["credits_note"] = SPECIAL_CREDITS_NOTE
            node["credits"] = 0 

    results = {
        "nodes": all_nodes,
        "relationships": []
    }

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{len(all_nodes)}개 노드 생성됨")


# --- 메인 실행 ---
if __name__ == "__main__":
    try:
        with open(INPUT, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
    except FileNotFoundError:
        exit(f"파일 없음: {INPUT}")

    create_subject_nodes(chunks)