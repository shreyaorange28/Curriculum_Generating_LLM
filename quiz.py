import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Set OpenAI API Key
load_dotenv(dotenv_path="../.env", override=True)

# Initialize LLM with higher temperature for variety
llm = ChatOpenAI(temperature=0.8)

# Prompt Template
quiz_prompt = PromptTemplate(
    input_variables=["topic", "difficulty", "question_number"],
    template="""
You are a helpful AI tutor. Write one UNIQUE question about "{topic}" at a "{difficulty}" difficulty level.
This is question #{question_number} - make sure it's different from previous questions.

For EASY level: Basic definitions, simple concepts, straightforward applications
For MEDIUM level: More complex relationships, multi-step thinking, analysis
For HARD level: Advanced concepts, synthesis, critical thinking, complex scenarios

Use this EXACT format:

Question: <actual question here>

A. <option>
B. <option>
C. <option>
D. <option>

**Correct Answer: X** (where X is the letter A, B, C, or D of the correct choice)

Explanation: <one sentence explaining why this answer is correct>

IMPORTANT: 
- Vary which letter (A, B, C, or D) is correct - don't always make A correct
- Make the question unique and appropriate for the {difficulty} level
- Ensure all 4 options are plausible but only one is clearly correct
"""
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_prompt)
difficulty_levels = ["easy", "medium", "hard"]

st.set_page_config(page_title="Adaptive Quiz", page_icon="📘")
st.title("📘 Quiz Tutor")

# Initialize session state
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "medium"
if "question_data" not in st.session_state:
    st.session_state.question_data = {}
if "question_number" not in st.session_state:
    st.session_state.question_number = 1
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 1
if "score" not in st.session_state:
    st.session_state.score = 0
if "selected" not in st.session_state:
    st.session_state.selected = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "quiz_finished" not in st.session_state:
    st.session_state.quiz_finished = False
if "pending_next" not in st.session_state:
    st.session_state.pending_next = False
if "question_generated" not in st.session_state:
    st.session_state.question_generated = False

# Parse function
def parse_question(raw):
    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    question = ""
    choices = []
    explanation = ""
    correct_choice = ""

    for line in lines:
        if line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
        elif line.startswith(("A.", "B.", "C.", "D.")):
            choices.append(line.strip())
        elif line.startswith("**Correct Answer:") and line.endswith("**"):
            # Extract the letter from "**Correct Answer: X**"
            correct_choice = line.replace("**Correct Answer:", "").replace("**", "").strip()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()

    if not correct_choice:
        for line in lines:
            if line.startswith(("A.", "B.", "C.", "D.")) and "**" in line:
                correct_choice = line.replace("**", "").strip().split(".")[0].strip()
                break

    # st.write("Debug - Raw AI Response:")
    # st.code(raw)
    # st.write(f"Debug - Parsed correct choice: '{correct_choice}'")
    # st.write(f"Debug - All choices: {choices}")
    
    return {
        "question": question,
        "choices": choices,
        "correct": correct_choice,
        "explanation": explanation
    }

# Step 1: Topic input
if not st.session_state.topic:
    topic_input = st.text_input("Enter a topic to begin:", "Newton's Laws")
    number_input = st.number_input("Enter number of questions", 1, 10)
    difficulty = st.selectbox("Enter a difficulty level:", difficulty_levels)
    if st.button("Start Quiz"):
        st.session_state.topic = topic_input
        st.session_state.total_questions = number_input
        st.session_state.question_number = 1 
        st.session_state.difficulty = difficulty
        st.session_state.question_generated = False



# Step 2: Generate a question (only once per question)
if (st.session_state.topic and 
    not st.session_state.question_generated and 
    not st.session_state.quiz_finished):
    
    with st.spinner("Generating question..."):
        result = quiz_chain.run({
            "topic": st.session_state.topic,
            "difficulty": st.session_state.difficulty,
            "question_number": st.session_state.question_number
        })
        st.session_state.question_data = parse_question(result)
        st.session_state.selected = None
        st.session_state.submitted = False
        st.session_state.pending_next = False
        st.session_state.question_generated = True


# Step 3: Display question and choices
if st.session_state.question_data and not st.session_state.quiz_finished:
    q = st.session_state.question_data
    st.markdown(f"### Q{st.session_state.question_number}: {q['question']}")
    
    # Display current difficulty level
    st.info(f"Current difficulty: {st.session_state.difficulty.title()}")
    
    # Create options for radio buttons
    options = [c.split(".")[0].strip() for c in q["choices"]]
    choice_map = {c.split(".")[0].strip(): c for c in q["choices"]}

    # Only show radio buttons if not yet submitted
    if not st.session_state.submitted:
        selected = st.radio(
            "Choose your answer:",
            options=options,
            format_func=lambda x: choice_map[x],
            index=None,
            key=f"radio_q{st.session_state.question_number}"
        )

        if st.button("Submit Answer", key=f"submit_{st.session_state.question_number}"):
            if selected:
                st.session_state.selected = selected
                st.session_state.submitted = True
                st.session_state.pending_next = True
                st.rerun()
            else:
                st.warning("Please select an answer before submitting.")
    
    # Show results after submission 
    if st.session_state.submitted and "result_processed" not in st.session_state:
        is_correct = st.session_state.selected == q["correct"]
        
        if is_correct:
            st.session_state.score += 1
            idx = difficulty_levels.index(st.session_state.difficulty)
            if idx < len(difficulty_levels) - 1:
                st.session_state.difficulty = difficulty_levels[idx + 1]
        else:
            idx = difficulty_levels.index(st.session_state.difficulty)
            if idx > 0:
                st.session_state.difficulty = difficulty_levels[idx - 1]
        
        st.session_state.result_processed = True
    
    # Display results 
    if st.session_state.submitted:
        st.markdown("**Answer choices:**")
        for choice in q["choices"]:
            choice_letter = choice.split(".")[0].strip()
            if choice_letter == q["correct"]:
                st.markdown(f"✅ **{choice}** ← Correct Answer")
            elif choice_letter == st.session_state.selected:
                st.markdown(f"❌ {choice} ← Your Answer")
            else:
                st.markdown(f"   {choice}")
        
        # Show result message
        if st.session_state.selected == q["correct"]:
            st.success(f"✅ Correct! Great job!")
            if hasattr(st.session_state, 'result_processed'):
                st.info(f"Difficulty increased to: {st.session_state.difficulty.title()}")
        else:
            st.error(f"❌ Incorrect. The correct answer was {q['correct']}.")
            if hasattr(st.session_state, 'result_processed'):
                st.info(f"Difficulty decreased to: {st.session_state.difficulty.title()}")

        st.info(f"💡 **Explanation:** {q['explanation']}")
        
        # Show current score
        st.markdown(f"**Current Score:** {st.session_state.score} / {st.session_state.question_number}")

# Step 4: Navigation and finish
if st.session_state.submitted and st.session_state.pending_next:
    if st.session_state.question_number >= st.session_state.total_questions:
        if st.button("Finish Quiz", key=f"finish_{st.session_state.question_number}"):
            st.session_state.quiz_finished = True
            st.rerun()
    else:
        if st.button("Next Question", key=f"next_{st.session_state.question_number}"):
            st.session_state.question_number += 1
            st.session_state.question_data = {}
            st.session_state.submitted = False
            st.session_state.pending_next = False
            st.session_state.question_generated = False  # Allow new question generation
            # Clear the result processing flag
            if "result_processed" in st.session_state:
                del st.session_state.result_processed
            st.rerun()

# Step 5: Final score
if st.session_state.quiz_finished:
    st.markdown("## 🎉 Quiz Complete!")
    st.markdown(f"**Final Score:** {st.session_state.score} / {st.session_state.total_questions}")
    percentage = (st.session_state.score / st.session_state.total_questions) * 100

    if percentage >= 80:
        st.balloons()
        st.success(f"Excellent! You scored {percentage:.1f}%")
    elif percentage >= 60:
        st.success(f"Good job! You scored {percentage:.1f}%")
    else:
        st.info(f"You scored {percentage:.1f}%. Keep practicing!")
    
    if st.button("Restart Quiz"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()