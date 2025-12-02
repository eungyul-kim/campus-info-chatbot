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

    [data-testid="stSidebar"] {
        background-color: #F2F2F2; 
    }

    h1 {
        font-size: 30px !important;
    }

    h2 {
        font-size: 20px !important;  
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

# 챗봇 초기화 (한 번만 실행)
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
    
# --- 3. 대화 기록 표시 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- 4. 사용자 입력 처리 ---
if prompt := st.chat_input("질문을 입력하세요..."):
    #llm에 보낼 history(현재 질문 제외)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"]
        if m["role"] in ("user", "assistant")
    ]

    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇 답변 생성
    with st.chat_message("assistant"):
        with st.spinner(f"답변 생성 중..."):
            
            response, chunks = rag_chatbot.chat(
                admission_year=admission_year, 
                department=department, 
                query=prompt,
                history=history,
                major_type=major_type 
            )
            
            # 답변 출력
            st.markdown(response)

    # 기록 저장
    st.session_state.messages.append({"role": "assistant", "content": response})