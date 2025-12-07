import google.generativeai as genai
import json
import os
import re
from dotenv import load_dotenv 

load_dotenv() 

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"[오류] {e}")
    exit()

INPUT_SUBSTITUTE_FILE = "output/substitutes_tables.json" 
INPUT_SUBJECTS_FILE = "output/subject_nodes.json" 
OUTPUT_REL_FILE = "output/substitutes_relationships.json" 
OUTPUT_NODE_FILE = "output/new_subject_nodes.json" 


def is_real_id(text):
    if not text: return False
    return bool(re.search(r'[A-Za-z]', text) and re.search(r'[0-9]', text))

# 노드 모양을 기존 데이터와 똑같이 맞춤
def format_node_schema(node):
    credits_val = node.get('credits')
    try:
        if credits_val:
            credits_val = int(credits_val)
    except:
        pass 

    return {
        "id": node.get('id'),
        "type": "Subject",          
        "name": node.get('name'),
        "credits": credits_val,
        "credits_note": None        
    }


def build_substitute_prompt(chunk, subject_list):
    metadata = chunk['metadata']
    metadata_str = json.dumps(metadata, ensure_ascii=False, indent=2)
    table_data_string = chunk['table_data_as_string'] 
    subject_str = json.dumps(subject_list, ensure_ascii=False)

    prompt = f"""
    당신은 대학 학사 행정 데이터 분석가입니다.
    [입력 데이터]는 '교과목 변경/대체/폐지' 표입니다.
    
    [문맥 정보]:
    {metadata_str}

    [입력 데이터 (표)]:
    {table_data_string}

    [참고: 기존 과목 마스터 리스트]:
    {subject_str}

    [*** 작업 순서 및 규칙 ***]
    
    **1. 뭉쳐진 과목 분리 (가장 중요):**
       - "설계프로젝트ABCD" -> "설계프로젝트A", "설계프로젝트B", "설계프로젝트C", "설계프로젝트D"로 분리하여 각각 별도의 관계(row)로 만드세요.
       - "기초물리학1,2" -> "기초물리학1", "기초물리학2"로 분리하세요.
       - 쉼표(,)나 슬래시(/)로 구분된 과목들도 모두 분리하세요.

    **2. Source 과목 (대체하는 과목) 찾기:**
       - 표의 '현행', '대체', '신설' 등에 해당하는 과목
       - id는 [참고 리스트]의 id를 사용.

    **3. Target 과목 (대체되는 과목) 찾기:**
       - 표의 '구', '타학과' 등에 해당하는 과목
       - id 처리 규칙
            Step A: [참고 리스트]에 있으면 그 id 사용.
            Step B: 표에 학수번호가 있으면 그 번호 사용.
            Step C: 둘 다 없으면 교과목명을 id로 사용 (괄호 내용 제거).

    **4. 새 노드 생성:**
       - Target 과목이 [참고 리스트]에 없을 때만 생성.
       - 속성으로 `id`, `name`, `credits` 만 추출.
       - 학점이 없으면 Source 과목 학점을 복사하세요.

    [출력 JSON 형식]:
    {{
      "relationships": [
        {{ "source_id": "...", "target_id": "...", "department": "...", "year": 2025, "note": null }}
      ],
      "new_nodes": [
        {{ "id": "...", "name": "...", "credits": 3 }} 
      ]
    }}
    """
    return prompt

def run_substitute_execution(chunks, subject_nodes):
    subject_list_prompt = [{"name": n["name"], "id": n["id"]} for n in subject_nodes]
    valid_ids = set(n['id'] for n in subject_nodes) 
    
    # 이름 확인용 매핑 테이블 
    id_name_map = {n['id']: n['name'] for n in subject_nodes}

    generation_config = {"response_mime_type": "application/json", "temperature": 0.0}
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)
    
    all_relationships = [] 
    all_new_nodes = {} 
    
    print(f"\n[대체 과목 분석] 총 {len(chunks)}개 데이터 처리 시작...")

    for i, chunk in enumerate(chunks):
        dept = chunk['metadata'].get('department', 'Unknown')
        print(f"  ({i+1}/{len(chunks)}) {dept} 분석 중...")
        
        prompt = build_substitute_prompt(chunk, subject_list_prompt)
        
        try:
            response = model.generate_content(prompt)
            result = json.loads(response.text)
            
            raw_rels = result.get('relationships', [])
            raw_nodes = result.get('new_nodes', [])

            # 1. 노드 처리
            for node in raw_nodes:
                new_id = node.get('id')
                new_name = node.get('name')
                
                if new_id and (new_id not in valid_ids):
                    formatted_node = format_node_schema(node)
                    
                    # 새 노드가 생기면 이름 매핑에도 추가
                    id_name_map[new_id] = new_name 

                    # Case 1: id 승격
                    if is_real_id(new_id) and (new_name in all_new_nodes):
                        old_dummy_id = new_name
                        del all_new_nodes[old_dummy_id]
                        all_new_nodes[new_id] = formatted_node
                        
                        if old_dummy_id in id_name_map: del id_name_map[old_dummy_id]
                        id_name_map[new_id] = new_name
                        
                        print(f"    ✨ [ID 승격] {old_dummy_id} -> {new_id}")

                        for rel in all_relationships:
                            if rel['source_id'] == old_dummy_id: rel['source_id'] = new_id
                            if rel['target_id'] == old_dummy_id: rel['target_id'] = new_id
                    
                    # Case 2: 일반
                    else:
                        if new_id not in all_new_nodes:
                            all_new_nodes[new_id] = formatted_node
                        else:
                            existing = all_new_nodes[new_id]
                            if (not existing.get('credits') or existing['credits']==0) and formatted_node.get('credits'):
                                existing['credits'] = formatted_node.get('credits')

            # 2. 관계 수집
            for rel in raw_rels:
                s_id = rel.get('source_id')
                t_id = rel.get('target_id')

                # id가 둘 다 있고, 자기 자신을 대체하는 것이 아닌 경우 추가
                if s_id and t_id and (s_id != t_id):
                    rel['type'] = "SUBSTITUTES"
                    all_relationships.append(rel)
            
            print(f"    => 관계 {len(raw_rels)}개, 새 과목 {len(raw_nodes)}개")
            
        except Exception as e:
            print(f"    [오류] {e}")

    # 3. 저장 및 후처리
    unique_rels_map = {}
    
    for r in all_relationships:
        s_id = r['source_id']
        t_id = r['target_id']
        
        r['source_name'] = id_name_map.get(s_id, s_id) 
        r['target_name'] = id_name_map.get(t_id, t_id)

        key = (s_id, t_id, r['department'], r.get('year'))
        unique_rels_map[key] = r
    
    final_rels = list(unique_rels_map.values())
    final_new_nodes = list(all_new_nodes.values())

    with open(OUTPUT_REL_FILE, 'w', encoding='utf-8') as f:
        json.dump({"relationships": final_rels}, f, ensure_ascii=False, indent=2)
        
    with open(OUTPUT_NODE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"nodes": final_new_nodes}, f, ensure_ascii=False, indent=2)
    
    print("-" * 50)
    print(f"[완료] 총 {len(final_rels)}개 관계 저장")


if __name__ == "__main__":
    if os.path.exists(INPUT_SUBSTITUTE_FILE) and os.path.exists(INPUT_SUBJECTS_FILE):
        with open(INPUT_SUBSTITUTE_FILE, 'r', encoding='utf-8') as f: chunks = json.load(f)
        with open(INPUT_SUBJECTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            sub_nodes = data.get("nodes", []) if isinstance(data, dict) else data
        run_substitute_execution(chunks, sub_nodes)
    else:
        print(f"[오류] 파일 없음")
        print(f"Substitutes: {INPUT_SUBSTITUTE_FILE}")
        print(f"Subjects:    {INPUT_SUBJECTS_FILE}")