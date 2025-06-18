import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv(dotenv_path="../.env", override=True)
llm = ChatOpenAI(temperature=0.5)

curriculum_prompt = PromptTemplate(
    input_variables=["topic", "num_lessons", "level", "mistakes"],
    template="Create a structured curriculum with {num_lessons} lesson topics for the subject: {topic}. Return only the numbered list:\n1. ...\n2. ...\n3. ...\n etc. Make sure to take their level of understanding ({level}) and any learning challenges ({challenges}) into account when organizing the curriculum. Keep the curriculum concise with each lesson topic under 10 words."
)

lesson_prompt = PromptTemplate(
    input_variables=["topic", "level", "mistakes"],
    template="""
You are a helpful and engaging AI tutor.

Create a comprehensive personalized lesson based on:
- Lesson Title: {lesson}
- Broader Unit: {topic}
- Level: {level}
- Learning Challenges: {challenges} (adjust explanation style and/or use analogies)
- Mistakes: {mistakes} (address these directly with clarification and repetition)

The lesson should include:
1. Clear explanation of key concepts (about two paragraphs)
2. Two examples (basic and advanced)
3. 3-question multiple choice quiz

Use this format:
**Title:** [Title]

**Explanation:**
[Explanation]

**Example 1: Basic**
[...]

**Example 2: Advanced**
[...]

**Quiz:**

**Q1:** [Question]
A. ...
B. ...
C. ...
D. ...
**Answer:** [Correct]

**Q2:** [Question]
...
**Answer:** [Correct]

**Q3:** [Question]
...
**Answer:** [Correct]
""")

quiz_prompt = PromptTemplate(
    input_variables=["topic", "difficulty", "question_number"],
    template="""
You are a helpful AI tutor. Write one UNIQUE question about "{topic}" at a "{difficulty}" difficulty level.

Use this format:This is question #{question_number} - make sure it's different from previous questions and randomize the correct answer.

Question: <question>
A. <option>
B. <option>
C. <option>
D. <option>
**Correct Answer: X**
Explanation: <one sentence explanation>
"""
)

curriculum_chain = LLMChain(llm=llm, prompt=curriculum_prompt)
lesson_chain = LLMChain(llm=llm, prompt=lesson_prompt)
quiz_chain = LLMChain(llm=llm, prompt=quiz_prompt)

st.set_page_config(page_title="AI Learning Platform", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .stTextInput > div > div {
        background-color: #f8f9fa;
    }
    .stTextArea > div > div {
        background-color: #f8f9fa;
    }
    .lesson-title {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .quiz-section {
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    .score-display {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .sidebar-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .progress-text {
        color: #34495e;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .practice-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
        border-left: 4px solid #27ae60;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "curriculum" not in st.session_state:
    st.session_state.curriculum = []
if "lesson_index" not in st.session_state:
    st.session_state.lesson_index = 0
if "lesson_data" not in st.session_state:
    st.session_state.lesson_data = {}
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "completed_lessons" not in st.session_state:
    st.session_state.completed_lessons = set()
if "practice_questions" not in st.session_state:
    st.session_state.practice_questions = []

def parse_quiz(text):
    lines = text.split('\n')
    quiz_started = False
    questions = {}
    q_num = 0
    for line in lines:
        if line.strip().startswith("**Quiz:**"):
            quiz_started = True
            continue
        if quiz_started:
            if line.strip().startswith("**Q"):
                q_num += 1
                q = line.split(":**", 1)[-1].strip()
                questions[q_num] = {"question": q, "choices": [], "answer": ""}
            elif line.strip().startswith(("A.", "B.", "C.", "D.")):
                questions[q_num]["choices"].append(line.strip())
            elif line.strip().startswith("**Answer:**"):
                questions[q_num]["answer"] = line.strip().split("**Answer:**")[-1].strip()
    return questions

def generate_practice_questions(topic, level):
    questions = []
    for i in range(1, 6):
        result = quiz_chain.run({"topic": topic, "difficulty": level, "question_number": i})
        parts = result.strip().split("\n")
        q_text, choices, correct, explanation = "", [], "", ""
        for part in parts:
            if part.startswith("Question:"):
                q_text = part.replace("Question:", "").strip()
            elif part.startswith(("A.", "B.", "C.", "D.")):
                choices.append(part.strip())
            elif part.startswith("**Correct Answer:"):
                correct = part.replace("**Correct Answer:", "").replace("**", "").strip()
            elif part.startswith("Explanation:"):
                explanation = part.replace("Explanation:", "").strip()
        questions.append({"question": q_text, "choices": choices, "correct": correct, "explanation": explanation})
    return questions

def get_completion_progress():
    if not st.session_state.curriculum:
        return 0
    return (len(st.session_state.completed_lessons) / len(st.session_state.curriculum)) * 100

# Tab setup
tab1, tab2 = st.tabs(["Create Curriculum", "Learning Dashboard"])

with tab1:
    st.markdown("# Curriculum Generator")
    st.markdown("Create a personalized learning curriculum tailored to your needs.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        topic = st.text_input("Subject", "Physics")
        level = st.selectbox("Difficulty Level", ["Lacks Foundation", "Understands a Little", "Understands Somewhat", "Understands a Lot"])
    
    with col2:
        num_lessons = st.slider("Number of Lessons", min_value=3, max_value=15, value=5)
        mistakes = st.text_area("Common Mistakes/Difficult Topics", placeholder="Optional: List common mistakes or areas of difficulty...")
        challenges = st.text_area("Learning Challenges", placeholder="Optional: List any learning difficulties or challenges (ex. must be at a 3rd grade level)")
    
    if st.button("Generate Curriculum", type="primary", use_container_width=True):
        with st.spinner("Creating your personalized curriculum..."):
            if (level == "Lacks Foundation"):
                num_lessons = num_lessons + 3;
                mistakes = mistakes + (
                " The first three lessons should focus on the essential prerequisite knowledge "
                "a learner must understand before diving into the main subject. Ensure these lessons "
                "cover foundational concepts that are commonly missing or assumed prior to learning the topic.")
            result = curriculum_chain.run({"topic": topic, "num_lessons": num_lessons, "challenges": challenges, "level": level})
            st.session_state.curriculum = [line.split(". ", 1)[1] for line in result.strip().split("\n") if ". " in line]
            st.session_state.lesson_index = 0
            st.session_state.lesson_data = {}
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.completed_lessons = set()
            st.session_state.practice_questions = []
        st.success("Curriculum created successfully! Switch to the Learning Dashboard to begin.")
        st.rerun()

with tab2:
    # Sidebar for lessons and progress
    with st.sidebar:
        if st.session_state.curriculum:
            st.markdown('<div class="sidebar-header">Progress Overview</div>', unsafe_allow_html=True)
            
            progress = get_completion_progress()
            st.progress(progress / 100)
            st.markdown(f'<div class="progress-text">{len(st.session_state.completed_lessons)} of {len(st.session_state.curriculum)} lessons completed</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<div class="sidebar-header">Lessons</div>', unsafe_allow_html=True)
            
            for i, item in enumerate(st.session_state.curriculum):
                is_completed = i in st.session_state.completed_lessons
                is_current = i == st.session_state.lesson_index
                
                if is_completed:
                    button_color = "#27ae60"
                    text_color = "white"
                elif is_current:
                    button_color = "#3498db"
                    text_color = "white"
                else:
                    button_color = "#ecf0f1"
                    text_color = "#2c3e50"
                
                btn = st.button(f"Lesson {i+1}: {item}", key=f"btn_{i}")
                
                st.markdown(f"""
                <style>
                div[data-testid='stButton'][key='btn_{i}'] button {{
                    background-color: {button_color} !important;
                    color: {text_color} !important;
                    border: none !important;
                    width: 100% !important;
                    border-radius: 8px !important;
                    padding: 0.75rem 1rem !important;
                    font-weight: 500 !important;
                    transition: all 0.3s ease !important;
                }}
                div[data-testid='stButton'][key='btn_{i}'] button:hover {{
                    opacity: 0.9 !important;
                    transform: translateY(-1px) !important;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
                }}
                </style>
                """, unsafe_allow_html=True)
                
                if btn:
                    st.session_state.lesson_index = i
                    st.session_state.quiz_submitted = False
                    st.session_state.practice_questions = []
                    st.rerun()
        else:
            st.info("Create a curriculum first to see your lessons here.")

    # Main content area
    if st.session_state.curriculum:
        current_topic = st.session_state.curriculum[st.session_state.lesson_index]
        current_lesson_num = st.session_state.lesson_index + 1
        total_lessons = len(st.session_state.curriculum)
        
        st.markdown(f'<div class="lesson-title">Lesson {current_lesson_num}: {current_topic}</div>', unsafe_allow_html=True)

        if current_topic not in st.session_state.lesson_data:
            with st.spinner("Generating lesson content..."):
                if level == "Lacks Foundation" and current_lesson_num < 3:
                    mistakes = mistakes +  "ensure the explanation introduces and clearly explains any background ideas or terminology the learner must understand before continuing to later lessons. Do not assume prior knowledge, and provide gentle, beginner-friendly explanations when appropriate."
                result = lesson_chain.run({"lesson": current_topic, "topic": topic, "level": level, "mistakes": mistakes, "challenges": challenges})
                st.session_state.lesson_data[current_topic] = result
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False

        full_lesson = st.session_state.lesson_data[current_topic]
        main_content = full_lesson.split("**Quiz:")[0]
        st.markdown(main_content)

        # Quiz section
        st.markdown('<div class="quiz-section">', unsafe_allow_html=True)
        st.markdown("### Knowledge Check")
        
        quiz = parse_quiz(full_lesson)
        
        if not st.session_state.quiz_submitted:
            with st.form("quiz_form"):
                for q_num, q in quiz.items():
                    st.markdown(f"**Question {q_num}:** {q['question']}")
                    options = [c.split(".", 1)[0].strip() for c in q["choices"]]
                    option_map = {c.split(".", 1)[0].strip(): c for c in q["choices"]}
                    st.session_state.quiz_answers[q_num] = st.radio(
                        "Select your answer:", options, format_func=lambda x: option_map[x], key=f"q{q_num}"
                    )
                    st.markdown("---")
                
                submitted = st.form_submit_button("Submit Quiz", type="primary")
                if submitted:
                    st.session_state.quiz_submitted = True
                    # Mark lesson as completed when quiz is submitted
                    st.session_state.completed_lessons.add(st.session_state.lesson_index)
                    st.rerun()
        else:
            score = 0
            for q_num, q in quiz.items():
                correct = q["answer"]
                selected = st.session_state.quiz_answers.get(q_num, "")
                st.markdown(f"**Question {q_num}:** {q['question']}")
                
                if selected == correct[0]:
                    st.success(f"Correct! Answer: {correct}")
                    score += 1
                else:
                    st.error(f"Incorrect. You selected: {selected} | Correct answer: {correct}")
                st.markdown("---")
            
            st.markdown(f'<div class="score-display">Final Score: {score} out of 3</div>', unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Retake Quiz", type="secondary"):
                    st.session_state.quiz_submitted = False
                    st.rerun()
            
            with col2:
                if st.button("Regenerate Lesson"):
                    with st.spinner("Regenerating lesson..."):
                        result = lesson_chain.run({"topic": current_topic, "level": level, "mistakes": mistakes})
                        st.session_state.lesson_data[current_topic] = result
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.practice_questions = []
                    st.rerun()
            
            with col3:
                if st.button("Extra Practice"):
                    with st.spinner("Generating practice questions..."):
                        st.session_state.practice_questions = generate_practice_questions(current_topic, level)
                    st.rerun()
            
            # Practice questions section
            if st.session_state.practice_questions:
                st.markdown('<div class="practice-section">', unsafe_allow_html=True)
                st.markdown("### Extra Practice Questions")
                for i, q in enumerate(st.session_state.practice_questions):
                    st.markdown(f"**Practice Question {i+1}:** {q['question']}")
                    for c in q['choices']:
                        st.markdown(c)
                    st.info(f"**Correct Answer:** {q['correct']} - {q['explanation']}")
                    st.markdown("---")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Next lesson button
            if st.session_state.lesson_index + 1 < len(st.session_state.curriculum):
                if st.button("Continue to Next Lesson", type="primary", use_container_width=True):
                    st.session_state.lesson_index += 1
                    st.session_state.quiz_submitted = False
                    st.session_state.practice_questions = []
                    st.rerun()
            else:
                completion_rate = get_completion_progress()
                if completion_rate == 100:
                    st.success("Congratulations! You have completed the entire curriculum.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        # Learning Dashboard
        
        No curriculum has been created yet. Please go to the "Create Curriculum" tab to get started.
        
        ### Once you create a curriculum, you'll see:
        - **Progress tracking** in the sidebar
        - **Interactive lessons** with comprehensive content
        - **Knowledge check quizzes** to test your understanding  
        - **Extra practice questions** for additional reinforcement
        - **Lesson regeneration** for alternative explanations
        """)