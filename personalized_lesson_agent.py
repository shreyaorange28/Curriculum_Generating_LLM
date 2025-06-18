# run with streamlit run personalized_lesson_agent.py
# python -m pip install streamlit langchain openai python-dotenv

import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Set OpenAI API Key
load_dotenv(dotenv_path="../.env", override=True)

# Initialize LLM
llm = ChatOpenAI(temperature=0.7)

# Prompt for lessons
lesson_prompt = PromptTemplate(
    input_variables=["topic", "level", "mistakes"],
    template="""
You are a helpful and engaging AI tutor.

Create a comprehensive personalized lesson based on the following:
- Topic: {topic}
- Level: {level}
- Common Mistakes: {mistakes}

The lesson should include:
1. A clear explanation of the topic with key concepts
2. Two illustrative examples (one basic, one advanced)
3. A short 3-question quiz with multiple choice answers based on the given explanation and examples

Format your response EXACTLY like this:

**Title:** [Lesson Title]

**Explanation:**
[Detailed explanation of the topic, highlighting key concepts and addressing common mistakes]

**Example 1: Basic**
[Simple, easy-to-understand example]

**Example 2: Advanced**
[More complex example that builds on the basic one]

**Quiz:**

**Q1:** [Question 1]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
**Answer:** [Correct letter]

**Q2:** [Question 2]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
**Answer:** [Correct letter]

**Q3:** [Question 3]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
**Answer:** [Correct letter]
"""
)

# Prompt Template for adaptive quiz
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
- Vary which letter (A, B, C, or D) is correct
- Make the question unique and appropriate for the {difficulty} level
- Ensure only one answer is clearly correct
"""
)

# Create chains
lesson_chain = LLMChain(llm=llm, prompt=lesson_prompt)
quiz_chain = LLMChain(llm=llm, prompt=quiz_prompt)
difficulty_levels = ["easy", "medium", "hard"]

st.set_page_config(page_title="AI Training Agent")
st.title("Training Agent")

# Initialize session state
if "mode" not in st.session_state:
    st.session_state.mode = "input"  # input, lesson, quiz, adaptive_quiz
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "level" not in st.session_state:
    st.session_state.level = "beginner"
if "mistakes" not in st.session_state:
    st.session_state.mistakes = ""
if "lesson_content" not in st.session_state:
    st.session_state.lesson_content = ""
if "lesson_quiz_answers" not in st.session_state:
    st.session_state.lesson_quiz_answers = {}
if "lesson_quiz_submitted" not in st.session_state:
    st.session_state.lesson_quiz_submitted = False

# Adaptive quiz session states
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

def parse_lesson_quiz(lesson_content):
    """Parse the quiz section from lesson content"""
    lines = lesson_content.split('\n')
    quiz_started = False
    current_question = None
    questions = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith("**Quiz:**"):
            quiz_started = True
            continue
        
        if quiz_started:
            if line.startswith("**Q") and ":**" in line:
                # Extract question number and text
                q_part = line.split(":**", 1)
                if len(q_part) == 2:
                    q_num = q_part[0].replace("**Q", "").strip()
                    q_text = q_part[1].strip()
                    current_question = q_num
                    questions[current_question] = {
                        "question": q_text,
                        "choices": [],
                        "answer": ""
                    }
            elif line.startswith(("A.", "B.", "C.", "D.")) and current_question:
                questions[current_question]["choices"].append(line)
            elif line.startswith("**Answer:**") and current_question:
                answer = line.replace("**Answer:**", "").strip()
                questions[current_question]["answer"] = answer
    
    return questions

def parse_adaptive_question(raw):
    """Parse adaptive quiz question"""
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
            correct_choice = line.replace("**Correct Answer:", "").replace("**", "").strip()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()

    if not correct_choice:
        for line in lines:
            if line.startswith(("A.", "B.", "C.", "D.")) and "**" in line:
                correct_choice = line.replace("**", "").strip().split(".")[0].strip()
                break
    
    return {
        "question": question,
        "choices": choices,
        "correct": correct_choice,
        "explanation": explanation
    }

# MODE 1: INPUT FORM
if st.session_state.mode == "input":
    st.markdown("## Let's Create Your Personalized Lesson!")
    
    with st.form("lesson_form"):
        topic = st.text_input("What topic would you like to learn about?", 
                             placeholder="e.g., Newton's Laws, Photosynthesis, Python Functions")
        level = st.selectbox("What's your current level?", 
                           ["beginner", "intermediate", "advanced"])
        mistakes = st.text_area("What concepts do you find confusing or want to focus on?", 
                               placeholder="e.g., confusing inertia with force, understanding when to use different methods")
        
        submitted = st.form_submit_button("Generate My Lesson!")
        
        if submitted and topic:
            st.session_state.topic = topic
            st.session_state.level = level
            st.session_state.mistakes = mistakes
            st.session_state.mode = "lesson"
            st.rerun()

# MODE 2: DISPLAY LESSON
elif st.session_state.mode == "lesson":
    if not st.session_state.lesson_content:
        with st.spinner("Creating your personalized lesson..."):
            lesson_content = lesson_chain.run({
                "topic": st.session_state.topic,
                "level": st.session_state.level,
                "mistakes": st.session_state.mistakes
            })
            st.session_state.lesson_content = lesson_content
    
    # Display lesson content
    lesson_words = st.session_state.lesson_content.split("**Quiz:")[0]
    st.markdown(lesson_words)
    
    # Parse and display quiz
    lesson_quiz = parse_lesson_quiz(st.session_state.lesson_content)
    
    if lesson_quiz and not st.session_state.lesson_quiz_submitted:
        st.markdown("---")
        st.markdown("## 📝 Quick Check Quiz")
        
        with st.form("lesson_quiz_form"):
            answers = {}
            for q_num, q_data in lesson_quiz.items():
                st.markdown(f"**Question {q_num}:** {q_data['question']}")
                
                # Create radio options
                choices = q_data['choices']
                choice_labels = [choice.split('.', 1)[1].strip() for choice in choices]
                choice_letters = [choice.split('.', 1)[0].strip() for choice in choices]
                
                selected = st.radio(
                    f"Choose your answer for Question {q_num}:",
                    options=choice_letters,
                    format_func=lambda x, choices=choices: next(c for c in choices if c.startswith(x)),
                    key=f"lesson_q{q_num}"
                )
                answers[q_num] = selected
                st.markdown("")
            
            quiz_submitted = st.form_submit_button("Submit Quiz")
            
            if quiz_submitted:
                st.session_state.lesson_quiz_answers = answers
                st.session_state.lesson_quiz_submitted = True
                st.rerun()
    
    # Show quiz results
    if st.session_state.lesson_quiz_submitted and lesson_quiz:
        st.markdown("---")
        st.markdown("## 📊 Quiz Results")
        
        correct_count = 0
        total_questions = len(lesson_quiz)
        
        for q_num, q_data in lesson_quiz.items():
            st.markdown(f"**Question {q_num}:** {q_data['question']}")
            user_answer = st.session_state.lesson_quiz_answers.get(q_num, "")
            correct_answer = q_data['answer']
            is_correct = user_answer == correct_answer[0]
            
            if is_correct:
                correct_count += 1
                st.success(f"✅ Question {q_num}: Correct! The answer was {correct_answer}.")
            else:
                st.error(f"❌ Question {q_num}: Incorrect. The correct answer was {correct_answer}. You chose {user_answer}.")
        
        percentage = (correct_count / total_questions) * 100
        st.info(f"**Your Score: {correct_count}/{total_questions} ({percentage:.1f}%)**")
        
        if percentage >= 80:
            st.success("🎉 Excellent work! You've mastered this topic!")
        elif percentage >= 67:
            st.warning("📈 Good job! You might benefit from some extra practice.")
        else:
            st.info("📚 Consider reviewing the lesson and trying some practice questions.")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 New Topic"):
            # Reset everything
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("🔄 Regenerate Lesson"):
            st.session_state.lesson_content = ""
            st.session_state.lesson_quiz_submitted = False
            st.session_state.lesson_quiz_answers = {}
            st.rerun()
    
    with col3:
        if st.button("💪 Extra Practice Questions"):
            st.session_state.mode = "adaptive_quiz"
            # Reset adaptive quiz states
            st.session_state.question_number = 1
            st.session_state.score = 0
            st.session_state.difficulty = st.session_state.level if st.session_state.level in ["easy", "medium", "hard"] else "medium"
            st.session_state.submitted = False
            st.session_state.quiz_finished = False
            st.session_state.question_generated = False
            st.session_state.question_data = {}
            st.rerun()

# MODE 3: ADAPTIVE QUIZ SETUP
elif st.session_state.mode == "adaptive_quiz" and st.session_state.total_questions == 1:
    st.markdown("## 💪 Extra Practice Questions")
    st.markdown(f"**Topic:** {st.session_state.topic}")
    
    with st.form("adaptive_quiz_setup"):
        st.markdown("Configure your practice session:")
        num_questions = st.number_input("How many practice questions?", 
                                      min_value=1, max_value=20, value=5)
        starting_difficulty = st.selectbox("Starting difficulty level:", 
                                         difficulty_levels, 
                                         index=difficulty_levels.index(st.session_state.difficulty))
        
        start_practice = st.form_submit_button("Start Practice! 🎯")
        
        if start_practice:
            st.session_state.total_questions = num_questions
            st.session_state.difficulty = starting_difficulty
            st.session_state.question_number = 1
            st.rerun()
    
    if st.button("← Back to Lesson"):
        st.session_state.mode = "lesson"
        st.rerun()

# MODE 4: ADAPTIVE QUIZ QUESTIONS
elif st.session_state.mode == "adaptive_quiz" and not st.session_state.quiz_finished:
    # Generate question if needed
    if not st.session_state.question_generated:
        with st.spinner("Generating practice question..."):
            result = quiz_chain.run({
                "topic": st.session_state.topic,
                "difficulty": st.session_state.difficulty,
                "question_number": st.session_state.question_number
            })
            st.session_state.question_data = parse_adaptive_question(result)
            st.session_state.selected = None
            st.session_state.submitted = False
            st.session_state.pending_next = False
            st.session_state.question_generated = True
    
    # Display question
    if st.session_state.question_data:
        q = st.session_state.question_data
        
        # Progress indicator
        progress = st.session_state.question_number / st.session_state.total_questions
        st.progress(progress)
        st.markdown(f"**Practice Question {st.session_state.question_number} of {st.session_state.total_questions}**")
        
        st.markdown(f"### {q['question']}")
        st.info(f"Current difficulty: {st.session_state.difficulty.title()}")
        
        # Create options for radio buttons
        options = [c.split(".")[0].strip() for c in q["choices"]]
        choice_map = {c.split(".")[0].strip(): c for c in q["choices"]}

        # Show radio buttons if not submitted
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
        
        # Process and show results
        if st.session_state.submitted and "result_processed" not in st.session_state:
            is_correct = st.session_state.selected == q["correct"]
            
            if is_correct:
                st.session_state.score += 1
                # Increase difficulty
                idx = difficulty_levels.index(st.session_state.difficulty)
                if idx < len(difficulty_levels) - 1:
                    st.session_state.difficulty = difficulty_levels[idx + 1]
            else:
                # Decrease difficulty
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
                st.success("✅ Correct! Great job!")
                if hasattr(st.session_state, 'result_processed'):
                    st.info(f"Difficulty increased to: {st.session_state.difficulty.title()}")
            else:
                st.error(f"❌ Incorrect. The correct answer was {q['correct']}.")
                if hasattr(st.session_state, 'result_processed'):
                    st.info(f"Difficulty decreased to: {st.session_state.difficulty.title()}")

            st.info(f"💡 **Explanation:** {q['explanation']}")
            st.markdown(f"**Current Score:** {st.session_state.score} / {st.session_state.question_number}")

        # Navigation
        if st.session_state.submitted and st.session_state.pending_next:
            col1, col2 = st.columns(2)
            
            if st.session_state.question_number >= st.session_state.total_questions:
                with col1:
                    if st.button("Finish Practice", key=f"finish_{st.session_state.question_number}"):
                        st.session_state.quiz_finished = True
                        st.rerun()
                with col2:
                    if st.button("← Back to Lesson"):
                        st.session_state.mode = "lesson"
                        st.rerun()
            else:
                with col1:
                    if st.button("Next Question →", key=f"next_{st.session_state.question_number}"):
                        st.session_state.question_number += 1
                        st.session_state.question_data = {}
                        st.session_state.submitted = False
                        st.session_state.pending_next = False
                        st.session_state.question_generated = False
                        if "result_processed" in st.session_state:
                            del st.session_state.result_processed
                        st.rerun()
                with col2:
                    if st.button("← Back to Lesson"):
                        st.session_state.mode = "lesson"
                        st.rerun()

# MODE 5: ADAPTIVE QUIZ RESULTS
elif st.session_state.mode == "adaptive_quiz" and st.session_state.quiz_finished:
    st.markdown("## 🎉 Practice Session Complete!")
    st.markdown(f"**Final Score:** {st.session_state.score} / {st.session_state.total_questions}")
    
    percentage = (st.session_state.score / st.session_state.total_questions) * 100
    
    if percentage >= 80:
        st.balloons()
        st.success(f"Outstanding! You scored {percentage:.1f}%")
        st.markdown("🌟 You've really mastered this topic!")
    elif percentage >= 60:
        st.success(f"Great work! You scored {percentage:.1f}%")
        st.markdown("📈 You're making excellent progress!")
    else:
        st.info(f"You scored {percentage:.1f}%. Keep practicing!")
        st.markdown("💪 Every expert was once a beginner. Keep going!")
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back to Lesson"):
            st.session_state.mode = "lesson"
            st.rerun()
    
    with col2:
        if st.button("🔄 More Practice"):
            # Reset for another practice session
            st.session_state.total_questions = 1  # This will trigger setup mode
            st.session_state.question_number = 1
            st.session_state.score = 0
            st.session_state.submitted = False
            st.session_state.quiz_finished = False
            st.session_state.question_generated = False
            st.session_state.question_data = {}
            st.rerun()
    
    with col3:
        if st.button("🏠 New Topic"):
            # Reset everything
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()