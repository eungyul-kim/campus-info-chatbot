import google.generativeai as genai
import json
import os
from dotenv import load_dotenv 
from collections import defaultdict

load_dotenv() 

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"[오류] API 키 설정 실패: {e}")
    exit()

INPUT = "output/includes_tables.json" 
SUB_NODES = "output/subject_nodes.json"
REQ_NODES = "output/requirement_nodes.json"
OUTPUT = "output/includes_relationships.json" 

def build_prompt(chunk, subject_list, req_id):
    metadata = json.dumps(chunk['metadata'], ensure_ascii=False, indent=2)
    table_string = chunk['table_data_as_string'] 
    subject = json.dumps(subject_list, ensure_ascii=False)

    prompt = f"""
    [입력 데이터]를 분석하여, `Requirement` 노드("{req_id}")와 `Subject` 노드 간의 `INCLUDES` 관계를 생성하세요.
    표에 적힌 '교과목명'을 보고 [참고 1] 리스트에서 정확한 `id`(학수번호)를 찾아내세요.

    [문맥 정보]:
    {metadata}

    [입력 데이터]:
    {table_string}

    [참고 1: 과목 노드 리스트 (ID, 이름, 별칭 포함)]:
    {subject}

    --- 작업 순서 ---
    1. [입력 데이터](표)에서 한 줄을 읽습니다. (예: `['전공필수 (42)', '객체지향프로그래밍, 웹/파이선...', '15']`)
    2. '구분' 열에서 '분류'(예: "전공필수", "전공선택", "전공기초")를 파악합니다. 
    3. 두 번째 열('교과목명')에서 '과목명들이 나열된 긴 문자열'을 찾습니다.
    4. 이 긴 문자열을 쉼표(`,`) 기준으로 모두 쪼개서 누락되는 과목이 없도록 과목명 리스트를 만듭니다. 
    5. 과목 id 검색 및 매칭
        - 과목명 뒤의 `(SWCON)`, `(EE)`, '(AI)' 같은 괄호는 제거하고, 순수한 과목명만 남깁니다.
        - 순수한 과목명으로 [참고 1] 리스트의 `name`을 검색하여 `id`를 찾습니다.
        - 만약 `name`에 없다면, [참고 1]의 `aliases`도 확인하여 일치하는 `id`를 찾습니다.
    6. 모든 관계에서 `source_id`는 졸업요건 노드의 id "{req_id}"로 고정합니다.
    
    --- 부전공 과정 유형이 여러 개인 경우 ---
    - 만약 표가 '부전공과정 1', '부전공과정 2' 혹은 '심화형', '인증형', '창업형' 등 여러 세부 유형으로 나뉘어 있다면,
      무조건 가장 첫 번째 유형만 취급하고 나머지 유형에 대한 정보는 무시하세요. 

    --- 동명이과목 처리 규칙 ---
    만약 이름(또는 별칭)이 같지만 id가 다른 과목이 여러 개 있다면, 현재 문맥정보(학과)에 맞춰 아래와 같이 선택하세요.
    - 컴퓨터공학과: `CSE`로 시작하는 ID 
    - 인공지능학과: `AI`로 시작하는 ID 
    - 소프트웨어융합학과: `SWCON`으로 시작하는 ID
    
    --- 속성 설정 규칙 ---
    - `classification`(메인 분류):
         - 오직 전공필수", "전공선택", "전공기초" 이 3가지만 허용됩니다.
         - 만약 '산학필수', '공통선택', '트랙필수' 등으로 확인된다면, 가장 상위 분류를 확인하여 위 3가지 중 하나로 입력하세요.
    - `sub_classification`(상세 분류):
         - 표의 하위구분에서 **"산학필수"**라고 명시된 경우에만 "산학필수"를 입력하세요.
         - 그 외의 경우('공통선택', '메타버스 분야' 등)는 무시하고 반드시 `null`로 처리하세요.


    반드시 아래 형식을 따르세요:

    {{
      "relationships": [
        {{
          "source_id": "{req_id}",
          "target_id": "CSE101", 
          "target_name_raw": "자료구조", 
          "classification": "전공필수",
          "sub_classification": null
        }}
      ]
    }}

    위 규칙대로 누락되는 것 없이 포함(include) 관계를 생성해 주세요.
    """
    return prompt

def create_includes_relationships(chunks, subject_nodes, requirement_nodes):
    # Requirement/Subject 노드를 이용해 INCLUDES 관계 생성

    # 1. 졸업요건 ID 맵핑: req_map[dept][year][track] = req_id
    req_map = defaultdict(lambda: defaultdict(dict))
    for req in requirement_nodes:
        req_id = req.get("id", "")
        parts = req_id.split("_")
        if len(parts) < 2:
            continue

        try:
            year = int(parts[0])
        except Exception:
            year = 2025

        dept = parts[1]
        target_tracks = ["단일전공", "다전공", "부전공"]

        for track in target_tracks:
            if track in req_id:
                req_map[dept][year][track] = req_id
                break

    # 2. 과목 리스트 최적화 (별칭 포함)
    optimized_subjects = []
    valid_ids = set()

    for n in subject_nodes:
        sid = n.get("id")
        if not sid:
            continue

        valid_ids.add(sid)
        entry = {"id": sid, "name": n.get("name")}
        if n.get("aliases"):
            entry["aliases"] = n["aliases"]
        optimized_subjects.append(entry)

    generation_config = {"response_mime_type": "application/json", "temperature": 0.0}
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

    all_relationships = []
    total = len(chunks)

    print(f"\n총 {total}개 청크 처리 시작")

    for i, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        dept = metadata.get("department", "")
        track = metadata.get("track", "")
        year = metadata.get("year", 2025)

        req_id = req_map[dept][year].get(track)

        if not req_id:
            continue

        print(f"{i + 1}/{total}: {year} {dept} [{track}] -> {req_id}")

        prompt = build_prompt(chunk, optimized_subjects, req_id)

        try:
            response = model.generate_content(prompt)
            raw = json.loads(response.text)
            raw_rels = raw.get("relationships", [])
        except Exception as e:
            print(f"  오류: {e}")
            continue

        best_rels = {}  # target_id 기준 병합용

        for rel in raw_rels:
            tid = rel.get("target_id")
            if not tid or tid not in valid_ids:
                continue

            # 분류/하위분류 정리
            current_class = (rel.get("classification") or "").replace(" ", "")
            current_sub = rel.get("sub_classification")

            rel["source_id"] = req_id
            rel["type"] = "INCLUDES"
            rel["classification"] = current_class

            if tid not in best_rels:
                best_rels[tid] = rel
            else:
                existing = best_rels[tid]

                # 여러 개의 분류에 포함될 경우 전공필수 우선
                if "필수" in existing.get("classification", ""):
                    final_class = existing["classification"]
                elif "필수" in current_class:
                    final_class = current_class
                else:
                    final_class = existing.get("classification", "")

                # 여러 개의 하위분류에 포함될 경우 산학필수 우선
                final_sub = existing.get("sub_classification") or current_sub

                existing["classification"] = final_class
                existing["sub_classification"] = final_sub

        valid_rels = list(best_rels.values())
        all_relationships.extend(valid_rels)
        print(f"  {len(valid_rels)}개 관계 추가")

    # source_id + target_id 기준 중복 제거
    unique_rels = list(
        {(r["source_id"], r["target_id"]): r for r in all_relationships}.values()
    )

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"nodes": [], "relationships": unique_rels}, f, ensure_ascii=False, indent=2)

    print(f"\n{len(unique_rels)}개 관계 저장됨: {OUTPUT}")


if __name__ == "__main__":
    if not os.path.exists(INPUT):
        exit("[오류] 입력 파일 없음")

    with open(INPUT, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(SUB_NODES, "r", encoding="utf-8") as f:
        sub_nodes = json.load(f).get("nodes", [])
    with open(REQ_NODES, "r", encoding="utf-8") as f:
        req_nodes = json.load(f).get("nodes", [])

    create_includes_relationships(chunks, sub_nodes, req_nodes)