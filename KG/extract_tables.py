"""
PDF에서 과목/졸업요건/대체 관련 표를 추출해 JSON으로 저장

- 입력: 처리할 PDF 경로와 페이지 범위가 담긴 JSON
- 출력: 표 데이터를 문자열로 담은 JSON
""" 

import json
import fitz 
import os

# 입출력 데이터는 생성할 노드/관계 종류에 따라 변경
INPUT = 'manifest/subject.json' 
OUTPUT = "output/subject_tables.json" 

def extract_table(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    results = []

    for pdf in manifest:
        path = pdf.get("file_path")
        common_meta = pdf.get("common_metadata", {})

        if not path or not os.path.exists(path):
            print(f"파일 없음: {path}")
            continue

        doc = fitz.open(path)
        print(f"파일 로드: {path}")
            
        sections = pdf.get('sections', [])

        for section in sections:
            section_meta = section.get('metadata', {})
            start = section.get('start_page') 
            end = section.get('end_page')     

            if start is None or end is None: continue

            department = section_meta.get('department')
            print(f"[section] {department} ({start}~{end})")
            
            all_tables = ""     # section의 모든 표 저장

            for p in range(start - 1, end):
                if p >= len(doc): continue
                    
                page = doc.load_page(p)
            
                tables = page.find_tables()     # 각 페이지의 모든 표 저장
                print(f" {p + 1}p: {len(tables.tables)}개 추출")

                # 각 표를 텍스트로 변환
                for i, table in enumerate(tables):
                    table_data = table.extract()
                    all_tables += f"\n--- {p + 1}p, table {i+1} ---\n"
                    all_tables += str(table_data) + "\n"

            if not all_tables:
                continue

            meta = {**common_meta, **section_meta}
            
            chunk = {
                "metadata": meta,
                "table_data_as_string": all_tables.strip()
            }
            results.append(chunk)
            
        doc.close()

    return results


if __name__ == "__main__":
    chunks = extract_table(INPUT)

    if chunks:
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"{OUTPUT} 저장됨")
