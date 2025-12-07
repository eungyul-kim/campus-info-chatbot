import streamlit as st
from backend import StreamlitRAGChatbot 

# --- 1. Streamlit 페이지 설정 ---
st.set_page_config(page_title="학사정보 챗봇", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
    
    .sidebar .stMarkdown > div > h2 {
        font-size: 1.5rem; 
        margin-top: 0;
        margin-bottom: 1rem;
        display: flex; 
        align-items: center;
    }

    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 20px !important;
        z-index: 1000 !important;
        left: 310px !important;    
        right: 50px !important;
    }

    [data-testid="stSidebar"] {
        background-color: #F2F2F2; 
    }

    h1 {
        font-size: 30px !important;
        margin-top: -40px !important;
        margin-bottom: 20px !important;
    }

    h2 {
        font-size: 20px !important;  
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;                  
        text-align: center;
        height: 40px;
        font-size: 32px;
        font-weight: bold;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 1px solid #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1>
        <i class="fa-solid fa-robot" style="color: #272F32; margin-right: 15px;"></i>
        학사정보 챗봇
    </h1>
    """,
    unsafe_allow_html=True
)

# 챗봇 초기화
@st.cache_resource(show_spinner="초기화 중...")
def initialize_chatbot():
    return StreamlitRAGChatbot()

rag_chatbot = initialize_chatbot()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요. 먼저 입학년도, 학과, 그리고 전공 유형을 설정해 주세요."}
    ]

# --- 2. 사용자 설정(Sidebar) ---
MIN_YEAR = 2020  
MAX_YEAR = rag_chatbot.LATEST_YEAR 
year_options = list(range(MIN_YEAR, MAX_YEAR + 1))[::-1] 

with st.sidebar:
    st.markdown(
        """
        <h2>
            <i class="fa-solid fa-gear" style="color: #272F32; margin-right: 7px;"></i> 
            학생 정보 설정
        </h2>
        """, 
        unsafe_allow_html=True
    )

    admission_year = st.selectbox(
        "입학년도",
        options=year_options,
        index=0, 
        key="admission_year_input_select",
        help="재학 중 개편된 교육과정으로 변경했다면, 변경한 연도를 선택하세요."
    )
    
    department = st.selectbox(
        "학과",
        options=rag_chatbot.get_departments(),
        key="department_select"
    )

    major_type = st.selectbox(
        "전공 유형",
        options=["단일전공", "다전공", "부전공"],
        index=0,
        key="major_type_select"
    )

# --- 3. 탭 구성 ---
tab1, tab2 = st.tabs(["## 💬 챗봇 상담", "## 🎓 졸업 자가진단"])

# --- 3-1. 챗봇 기능 ---
with tab1:
    
    # 질문입력란이 항상 화면 하단에 뜨도록 고정
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 입력한 내용을 container 안에
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    # history에서 방금 질문 제외
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1] 
                        if m["role"] in ("user", "assistant")
                    ]
                    
                    # Backend 호출
                    response, source = rag_chatbot.chat(
                        admission_year=admission_year, 
                        department=department, 
                        query=prompt,
                        history=history,
                        major_type=major_type 
                    )
                
                    # 답변 출력
                    st.markdown(response)
                    if source:
                        with st.expander("📚 출처 확인"):
                            for src in source:
                                # URL인 경우 
                                if src.get('url'):
                                    st.markdown(f"🌐 [**{src['name']}** 바로가기]({src['url']})")
                                # PDF 파일인 경우
                                else:
                                    display_text = f"📄 **{src['name']}**"
                                    st.markdown(display_text)

        # 기록 저장
        st.session_state.messages.append({"role": "assistant", "content": response})
        

# --- 3-2. 학점 계산기 ---
with tab2:
    st.markdown("##### 📝 수강한 과목을 입력하세요")
    st.caption("(쉼표, 줄바꿈으로 구분)")
    
    # 수강한 과목 입력받기
    taken_input = st.text_area(
        "과목 입력",
        placeholder="예시: 자료구조, 운영체제, 컴퓨터구조, 캡스톤디자인",
        height=150,
        label_visibility="collapsed"
    )
    
    if st.button("진단 시작", type="primary", use_container_width=True):
        if not taken_input:
            st.warning("수강한 과목을 입력해주세요!")
        else:
            with st.spinner("졸업요건 분석 중..."):
                # 입력값 리스트로 변환 (쉼표, 줄바꿈 제거)
                taken_list = [s.strip() for s in taken_input.replace('\n', ',').split(',') if s.strip()]
                
                # 계산 함수 호출
                req_info, missing_result, credit_status = rag_chatbot.check_graduation_status(
                    admission_year, department, major_type, taken_list
                )
                
                st.divider()

                # 남은 학점 수를 보여줌
                st.subheader("(1) 학점 이수 현황")
                col1, col2, col3, col4 = st.columns(4)
                
                def get_metric_data(cat_name):
                    data = credit_status.get(cat_name, {'required': 0, 'earned': 0, 'remaining': 0})
                    label = f"{data['earned']} / {data['required']}"
                    delta = f"-{data['remaining']} 학점" if data['remaining'] > 0 else "이수 완료!"
                    color = "normal" if data['remaining'] > 0 else "off"
                    return label, delta, color

                with col1:
                    l, d, c = get_metric_data('전공필수')
                    st.metric(label="전공필수", value=l, delta=d, delta_color="inverse")
                
                with col2:
                    l, d, c = get_metric_data('전공기초')
                    st.metric(label="전공기초", value=l, delta=d, delta_color="inverse")

                with col3:
                    l, d, c = get_metric_data('전공선택')
                    st.metric(label="전공선택", value=l, delta=d, delta_color="inverse")

                with col4:
                    l, d, c = get_metric_data('산학필수')
                    st.metric(label="산학필수", value=l, delta=d, delta_color="inverse")

                st.divider()

                
                # 남은 과목명을 보여줌(전공필수/기초만)
                st.subheader("(2) 미이수 과목")
                
                if not missing_result:
                    st.success("모든 필수 과목을 이수했습니다.")
                    st.balloons()
                else:
                    st.error("아직 이수하지 않은 필수 과목이 있습니다.")
                    
                    # 카테고리별로 확장해서 보여줌
                    for category, subjects in missing_result.items():
                        with st.expander(f"📌 {category} ({len(subjects)}건)", expanded=True):
                            for sub in subjects:

                                st.markdown(f"- **{sub['name']}** ({sub['credits']}학점)")
