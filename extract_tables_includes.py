"""
PDF에서 전공과목 편성표를 추출해 JSON으로 저장(include생성용)
메타데이터로 track(단일/다/부전공) 정보 추가

- 입력: 처리할 PDF 경로와 페이지 범위가 담긴 JSON
- 출력: 표 데이터를 문자열로 담은 JSON
""" 

import json
import fitz  
import os
import re

INPUT = 'manifest/includes.json' 
OUTPUT = "output/includes_tables.json" 

def detect_track(title: str, last_track: str | None) -> str | None:    
    '''
    표 제목에서 전공유형(단일/다/부)을 찾음
    표 제목 구조: [표1] 단일전공 전공과목 편성 -> [표N] 바로 뒤에 전공 유형이 명시돼 있음을 이용      
    '''
    current_track = None
    matches = list(re.finditer(r"\[\s*표\s*\d+\s*\]", title))

    if matches: # Case 1: [표N]이 있으면, [표N] 이후 텍스트만 검사
        last_match = matches[-1]
        target = title[last_match.end():]

        if "부전공" in target:
            current_track = "부전공"
        elif "다전공" in target:
            current_track = "다전공"
        elif "단일전공" in target:
            current_track = "단일전공"
        else:
            current_track = None  
    else:   # Case 2: [표N]이 없는 경우(표가 다음페이지까지 이어지는 경우)
        if len(title) < 2:
            current_track = last_track
        else:
            current_track = None  
     
    return current_track


def extract_includes(manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list):
        manifest = []

    results = []

    for pdf in manifest:
        path = pdf.get("file_path")
        common_meta = pdf.get("common_metadata", {})

        if not path or not os.path.exists(path):
            print(f"파일 없음: {path}")
            continue

        doc = fitz.open(path)
        print(f"파일 로드: {path}")

        for section in pdf.get("sections", []):
            section_meta = section.get("metadata", {})
            start = section.get("start_page")
            end = section.get("end_page")

            if start is None or end is None:
                continue

            last_track = None  # 다음페이지로 이어지는 표를 위한 상속값

            for p in range(start - 1, end):
                if p >= len(doc):
                    continue

                page = doc.load_page(p)
                tables = page.find_tables()

                if not tables.tables:
                    continue

                print(f" {p + 1}p: {len(tables.tables)}개 표")

                for i, table in enumerate(tables.tables):
                    # 1. 표 위 제목 추출 
                    x0, y0, x1, y1 = table.bbox
                    rect_above = fitz.Rect(
                        0,
                        max(0, y0 - 150),
                        page.rect.width,
                        y0,
                    )
                    raw_title = page.get_text("text", clip=rect_above)
                    title = raw_title.replace("\n", " ").strip()

                    # 2. 전공 유형 판별
                    track = detect_track(title, last_track)
                    if not track:
                        continue

                    last_track = track
                    print(f"   track: {track} (p{p + 1}, t{i + 1})")

                    # 3. 표 텍스트 추출 
                    table_data = table.extract()
                    text = json.dumps(table_data, ensure_ascii=False)
                    text = (
                        text.replace("\\u0001", " ")
                            .replace("\\n", " ")
                            .replace("null", '""')
                    )

                    # 4. 저장
                    meta = {**common_meta, **section_meta}
                    meta["page_number"] = p + 1
                    meta["table_index"] = i + 1
                    meta["track"] = track

                    chunk = {
                        "metadata": meta,
                        "table_data_as_string": text,
                    }
                    results.append(chunk)

        doc.close()

    return results


# ---------------------------------------------
# 실행
# ---------------------------------------------
if __name__ == "__main__":
    chunks = extract_includes(INPUT)

    if chunks:
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"{OUTPUT} 저장됨")