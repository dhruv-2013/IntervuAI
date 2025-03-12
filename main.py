import streamlit as st
import os
import tempfile
import numpy as np
import time
from pathlib import Path
from faster_whisper import WhisperModel
import openai
from datetime import datetime
from audio_recorder_streamlit import audio_recorder
from PIL import Image
import base64
from io import BytesIO
from google.cloud import texttospeech
import google.auth

# Set path to Google Cloud credentials file
GOOGLE_CREDENTIALS_PATH = "C:\\Users\\dhruv\\Desktop\\Interview_Bot\\durable-stack-453203-c6-c007f8e298d9.json"

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state variables
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_question_idx" not in st.session_state:
    st.session_state.current_question_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "use_gpu" not in st.session_state:
    st.session_state.use_gpu = False
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False
if "selected_job_field" not in st.session_state:
    st.session_state.selected_job_field = None
if "setup_stage" not in st.session_state:
    st.session_state.setup_stage = "job_selection"
if "question_spoken" not in st.session_state:
    st.session_state.question_spoken = False
if "use_voice" not in st.session_state:
    st.session_state.use_voice = True
if "interviewer_name" not in st.session_state:
    st.session_state.interviewer_name = ""
if "voice_type" not in st.session_state:
    st.session_state.voice_type = "en-US-Neural2-D"
# New variable to track interview stage
if "interview_stage" not in st.session_state:
    st.session_state.interview_stage = "introduction"

@st.cache_resource
def load_whisper_model():
    model_size = "small" if st.session_state.get("faster_transcription", True) else "medium"
    device = "cuda" if st.session_state.get("use_gpu", False) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)

# Initialize Google Cloud Text-to-Speech client
@st.cache_resource
def get_tts_client():
    try:
        # Explicitly use credentials file
        credentials, project = google.auth.load_credentials_from_file(GOOGLE_CREDENTIALS_PATH)
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        return client
    except Exception as e:
        st.error(f"Error initializing Google Cloud TTS client: {str(e)}")
        return None

# Function to generate speech from text using Google Cloud TTS
def text_to_speech(text):
    client = get_tts_client()
    if not client:
        raise Exception("Failed to initialize Google Cloud TTS client")
    
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=st.session_state.voice_type,
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    
    # Select the type of audio file
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.95,  # Slightly slower for interview questions
        pitch=0.0,  # Natural pitch
        volume_gain_db=1.0  # Slightly louder
    )
    
    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    # Return the audio content as a BytesIO object
    fp = BytesIO(response.audio_content)
    fp.seek(0)
    return fp

# Function to create an HTML audio player with autoplay for TTS
def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

st.set_page_config(
    page_title="Interview Agent",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
JOB_FIELDS = {
    "Software Engineering": {
        "Technical": [
            "Explain the difference between arrays and linked lists.",
            "What's your approach to debugging a complex issue?",
            "Describe a challenging technical problem you solved recently.",
            "Explain the concept of time and space complexity.",
            "What design patterns have you used in your projects?",
            "How do you ensure your code is maintainable and scalable?",
            "Explain how you would implement error handling in a distributed system."
        ],
        "Behavioral": [
            "Tell me about a time you had to work under pressure to meet a deadline.",
            "Describe a situation where you disagreed with a team member on a technical approach.",
            "How do you handle feedback on your code during code reviews?",
            "Tell me about a time you identified and fixed a bug that others couldn't solve.",
            "How do you keep up with the latest technologies and programming languages?"
        ],
        "Role-specific": [
            "How do you approach testing your code?",
            "Describe your experience with CI/CD pipelines.",
            "How do you balance technical debt with delivering features?",
            "Explain your approach to optimizing application performance.",
            "How would you explain a complex technical concept to a non-technical stakeholder?"
        ]
    },
    "Data Science/Analysis": {
        "Technical": [
            "Explain the difference between supervised and unsupervised learning.",
            "How do you handle missing data in a dataset?",
            "Describe a data cleaning process you've implemented.",
            "What statistical methods do you use to validate your findings?",
            "Explain the concept of overfitting and how to avoid it.",
            "How would you approach feature selection for a machine learning model?",
            "Explain the difference between correlation and causation with an example."
        ],
        "Behavioral": [
            "Tell me about a time when your data analysis led to a significant business decision.",
            "How do you communicate complex data insights to non-technical stakeholders?",
            "Describe a situation where you had to defend your analytical approach.",
            "Tell me about a project where you had to work with messy or incomplete data.",
            "How do you ensure your analysis is accurate and reliable?"
        ],
        "Role-specific": [
            "What visualization tools do you prefer and why?",
            "How do you determine which statistical test to use for a given problem?",
            "Describe your approach to A/B testing.",
            "How do you translate business questions into data queries?",
            "What metrics would you track to measure the success of a product feature?"
        ]
    },
    "Project Management": {
        "Technical": [
            "What project management methodologies are you familiar with?",
            "How do you create and maintain a project schedule?",
            "Describe your approach to risk management.",
            "How do you track and report project progress?",
            "What tools do you use for project planning and why?",
            "How do you handle resource allocation in a project?",
            "Explain how you would manage scope creep."
        ],
        "Behavioral": [
            "Tell me about a time when a project was falling behind schedule.",
            "Describe how you've managed stakeholder expectations.",
            "How do you motivate team members during challenging phases of a project?",
            "Tell me about a project that failed and what you learned from it.",
            "How do you handle conflicts between team members or departments?"
        ],
        "Role-specific": [
            "How do you prioritize competing deadlines across multiple projects?",
            "Describe how you communicate project status to different audiences.",
            "How do you ensure quality deliverables while maintaining timelines?",
            "What's your approach to gathering requirements from stakeholders?",
            "How do you manage project budgets and resources?"
        ]
    },
    "UX/UI Design": {
        "Technical": [
            "Walk me through your design process.",
            "How do you approach user research?",
            "Describe how you create and use personas.",
            "What tools do you use for wireframing and prototyping?",
            "How do you incorporate accessibility into your designs?",
            "Explain the importance of design systems.",
            "How do you use data to inform design decisions?"
        ],
        "Behavioral": [
            "Tell me about a time when you received difficult feedback on your design.",
            "Describe a situation where you had to compromise on a design decision.",
            "How do you advocate for the user when there are business constraints?",
            "Tell me about a design challenge you faced and how you overcame it.",
            "How do you collaborate with developers to implement your designs?"
        ],
        "Role-specific": [
            "How do you measure the success of a design?",
            "Describe how you stay current with design trends and best practices.",
            "How do you balance aesthetics with usability?",
            "Explain your approach to responsive design.",
            "How would you improve the user experience of our product?"
        ]
    },
    "IT Support": {
        "Technical": [
            "Explain the difference between hardware and software troubleshooting.",
            "How would you approach a user who can't connect to the internet?",
            "Describe your experience with ticketing systems.",
            "What steps would you take to secure a workstation?",
            "How do you prioritize multiple support requests?",
            "Explain how you would troubleshoot a slow computer."
        ],
        "Behavioral": [
            "Tell me about a time when you had to explain a technical issue to a non-technical user.",
            "Describe a situation where you went above and beyond for a user.",
            "How do you handle frustrated or angry users?",
            "Tell me about a time when you couldn't solve a technical problem immediately.",
            "How do you stay patient when dealing with repetitive support issues?"
        ],
        "Role-specific": [
            "What remote support tools are you familiar with?",
            "How do you document your troubleshooting steps?",
            "Describe your approach to user training and education.",
            "How do you keep up with new technologies and support techniques?",
            "What's your experience with supporting remote workers?"
        ]
    },
    "Cybersecurity": {
        "Technical": [
            "Explain the concept of defense in depth.",
            "What's the difference between authentication and authorization?",
            "How would you respond to a potential data breach?",
            "Describe common network vulnerabilities and how to mitigate them.",
            "What's your approach to vulnerability assessment?",
            "Explain the importance of patch management."
        ],
        "Behavioral": [
            "Tell me about a time when you identified a security risk before it became an issue.",
            "How do you balance security needs with user convenience?",
            "Describe a situation where you had to convince management to invest in security measures.",
            "How do you stay current with evolving security threats?",
            "Tell me about a time when you had to respond to a security incident."
        ],
        "Role-specific": [
            "What security tools and technologies are you experienced with?",
            "How would you implement a security awareness program?",
            "Describe your experience with compliance requirements (GDPR, HIPAA, etc.)",
            "What's your approach to security logging and monitoring?",
            "How would you conduct a security audit?"
        ]
    }
}

COMMON_QUESTIONS = {
    "Background": [
        "Tell me more about yourself and why you're interested in this field."
    ]
}

def generate_questions(job_field, categories, num_questions):
    questions = []
    job_categories = [cat for cat in categories if cat in JOB_FIELDS[job_field]]
    common_categories = [cat for cat in categories if cat in COMMON_QUESTIONS]
    
    # Define the question order sequence
    category_order = []
    
    # Background questions always come first
    if "Background" in common_categories:
        category_order.append("Background")
        common_categories.remove("Background")
    
    # Technical questions come next
    if "Technical" in job_categories:
        category_order.append("Technical")
        job_categories.remove("Technical")
    
    # Behavioral questions come after technical
    if "Behavioral" in job_categories:
        category_order.append("Behavioral")
        job_categories.remove("Behavioral")
    
    # Role-specific questions come last
    if "Role-specific" in job_categories:
        category_order.append("Role-specific")
        job_categories.remove("Role-specific")
    
    # Add any remaining category types
    category_order.extend(job_categories)
    category_order.extend(common_categories)
    
    # Start with at least one question from each category in the specified order
    for category in category_order:
        if category in JOB_FIELDS[job_field]:
            questions.append({
                "category": category,
                "question": np.random.choice(JOB_FIELDS[job_field][category])
            })
        elif category in COMMON_QUESTIONS:
            questions.append({
                "category": category,
                "question": np.random.choice(COMMON_QUESTIONS[category])
            })
    
    # Fill remaining slots (if needed)
    remaining_slots = num_questions - len(questions)
    if remaining_slots > 0:
        question_pool = []
        for category in category_order:
            if category in JOB_FIELDS[job_field]:
                category_questions = JOB_FIELDS[job_field][category]
            elif category in COMMON_QUESTIONS:
                category_questions = COMMON_QUESTIONS[category]
            else:
                continue
            
            # Add more questions from the same categories, maintaining the order
            for q in category_questions:
                if {"category": category, "question": q} not in questions:
                    question_pool.append({"category": category, "question": q})
        
        if question_pool:
            # Group by category to maintain order while selecting additional questions
            grouped_pool = {}
            for q in question_pool:
                if q["category"] not in grouped_pool:
                    grouped_pool[q["category"]] = []
                grouped_pool[q["category"]].append(q)
            
            additional_questions = []
            remaining = remaining_slots
            
            # Take additional questions from each category in order until we've filled the slots
            while remaining > 0 and any(len(grouped_pool.get(cat, [])) > 0 for cat in category_order):
                for cat in category_order:
                    if cat in grouped_pool and grouped_pool[cat]:
                        # Randomly select one question from this category
                        idx = np.random.randint(0, len(grouped_pool[cat]))
                        additional_questions.append(grouped_pool[cat].pop(idx))
                        remaining -= 1
                        if remaining == 0:
                            break
            
            questions.extend(additional_questions)
    
    # No need to shuffle since we want to maintain the category order
    return questions

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file)
        temp_audio_path = temp_audio.name
    
    model = load_whisper_model()
    
    if st.session_state.get("faster_transcription", True):
        segments, info = model.transcribe(
            temp_audio_path, 
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            language="en"
        )
    else:
        segments, info = model.transcribe(
            temp_audio_path, 
            beam_size=5,
            language="en"
        )
    
    transcript = ""
    for segment in segments:
        transcript += segment.text + " "
    
    os.unlink(temp_audio_path)
    
    return transcript.strip()

def get_answer_feedback(question, answer):
    prompt = f"""
    You are an expert interview coach. Analyze the following interview response and provide detailed, constructive feedback:
    
    Question: {question}
    Answer: {answer}
    
    Provide feedback with the following structure:
    1. Strength Assessment: What was strong about this answer?
    2. Area for Improvement: What could be improved?
    3. Missing Elements: What important points were missed?
    4. Communication Style: How was the delivery (clarity, conciseness, etc.)?
    5. Overall Score: Rate the answer from 1-10
    6. Improved Response: Provide a brief example of an improved answer
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert interview coach providing constructive feedback on interview responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

# Interview progress sidebar
if st.session_state.questions and st.session_state.setup_stage == "interview":
    with st.sidebar:
        st.title("Interview Progress")
        progress = (st.session_state.current_question_idx) / len(st.session_state.questions)
        st.progress(progress)
        st.write(f"Question {st.session_state.current_question_idx + 1} of {len(st.session_state.questions)}")
        
        if st.button("End Interview & See Results"):
            st.session_state.interview_complete = True
            st.rerun()
        
        if st.button("Restart Interview"):
            for key in ['questions', 'current_question_idx', 'answers', 'feedbacks', 
                       'recording', 'audio_data', 'transcription', 'interview_complete', 
                       'show_feedback', 'question_spoken']:
                if key in st.session_state:
                    if isinstance(st.session_state[key], list):
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = False
            st.session_state.current_question_idx = 0
            st.session_state.questions = []
            st.session_state.interview_stage = "introduction"
            st.session_state.setup_stage = "job_selection"
            st.rerun()

# Job selection screen
if st.session_state.setup_stage == "job_selection" and not st.session_state.questions:
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    #MainMenu {
        display: none;
    }
    footer {
        display: none;
    }
    header {
        display: none;
    }
    .stButton > button {
        background-color: white;
        color: black;
        border: none;
        border-radius: 4px;
        padding: 15px;
        font-size: 16px;
        font-weight: normal;
        text-align: left;
        margin: 10px 0;
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .stButton > button:hover {
        background-color: #f0f0f0;
    }
    .stButton > button::after {
        content: "›";
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 5px;"></div>', unsafe_allow_html=True)
    
    _, center_title_col, _ = st.columns([1, 3, 1])
    
    with center_title_col:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 48px; font-weight: normal; margin: 0; padding: 0; color: white;">
                Interview Agent
            </h1>
            <h2 style="font-size: 28px; font-weight: normal; margin: 0; padding: 0; color: white;">
                What field do you want to practice for?
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        for job_field in JOB_FIELDS.keys():
            if st.button(f"{job_field}", key=f"{job_field}Button", use_container_width=True):
                st.session_state.selected_job_field = job_field
                st.session_state.setup_stage = "category_selection"
                st.rerun()

# Category selection screen
elif st.session_state.setup_stage == "category_selection" and not st.session_state.questions:
    st.subheader("Select question categories and settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        job_specific_categories = list(JOB_FIELDS[st.session_state.selected_job_field].keys())
        selected_job_categories = []
        st.write("**Job-specific categories:**")
        for category in job_specific_categories:
            if st.checkbox(f"{category}", value=True, key=f"job_{category}"):
                selected_job_categories.append(category)
        
        # Always include Background
        st.write("**Common categories:**")
        st.checkbox("Background", value=True, key="common_Background", disabled=True)
        selected_categories = selected_job_categories + ["Background"]
    
    with col2:
        num_questions = st.slider("Number of questions:", 3, 15, 5)
        
        # Change label to clarify this is for the interviewee's name
        interviewee_name = st.text_input("Your name (interviewee):", value=st.session_state.interviewer_name)
        st.session_state.interviewer_name = interviewee_name
        
        st.session_state.use_voice = st.checkbox("Enable voice for questions", value=True)
        
        if st.session_state.use_voice:
            voice_options = {
                "Male (Default)": "en-US-Neural2-D", 
                "Female": "en-US-Neural2-F",
                "Male (British)": "en-GB-Neural2-B",
                "Female (British)": "en-GB-Neural2-C"
            }
            selected_voice = st.selectbox(
                "Select interviewer voice:",
                options=list(voice_options.keys()),
                index=0
            )
            st.session_state.voice_type = voice_options[selected_voice]
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", key="back_button"):
            st.session_state.setup_stage = "job_selection"
            st.session_state.selected_job_field = None
            st.rerun()
    
    with col3:
        if st.button("Start Practice →", type="primary", key="start_practice"):
            if selected_categories:
                st.session_state.questions = generate_questions(
                    st.session_state.selected_job_field, 
                    selected_categories, 
                    num_questions
                )
                st.session_state.current_question_idx = 0
                st.session_state.answers = [""] * len(st.session_state.questions)
                st.session_state.feedbacks = [""] * len(st.session_state.questions)
                st.session_state.interview_complete = False
                st.session_state.show_feedback = False
                st.session_state.question_spoken = False
                st.session_state.interview_stage = "introduction"
                st.session_state.setup_stage = "interview"
                st.rerun()
            else:
                st.error("Please select at least one category.")

# Interview results screen
elif st.session_state.interview_complete:
    st.title("Interview Practice Results")
    
    if st.session_state.answers and not all(answer == "" for answer in st.session_state.answers):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        session_data = {
            "timestamp": timestamp,
            "questions": st.session_state.questions,
            "answers": st.session_state.answers,
            "feedbacks": st.session_state.feedbacks
        }
        st.session_state.session_history.append(session_data)
    
    for i, (question_data, answer, feedback) in enumerate(zip(
            st.session_state.questions, 
            st.session_state.answers, 
            st.session_state.feedbacks)):
        
        with st.expander(f"Question {i+1}: {question_data['question']} ({question_data['category']})", expanded=i==0):
            st.write("**Your Answer:**")
            if answer:
                st.write(answer)
            else:
                st.write("*No answer provided*")
            
            st.write("**Feedback:**")
            if feedback:
                st.write(feedback)
            else:
                if answer:
                    with st.spinner("Generating feedback..."):
                        feedback = get_answer_feedback(question_data['question'], answer)
                        st.session_state.feedbacks[i] = feedback
                        st.write(feedback)
                else:
                    st.write("*No feedback available (no answer provided)*")
    
    if st.button("Practice Again", type="primary"):
        for key in ['questions', 'current_question_idx', 'answers', 'feedbacks', 
                   'recording', 'audio_data', 'transcription', 'interview_complete',
                   'question_spoken']:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = False
        st.session_state.current_question_idx = 0
        st.session_state.questions = []
        st.session_state.interview_stage = "introduction"
        st.rerun()
    
    if st.session_state.session_history and st.button("View Practice History"):
        st.subheader("Your Practice History")
        for i, session in enumerate(reversed(st.session_state.session_history)):
            with st.expander(f"Session {len(st.session_state.session_history) - i}: {session['timestamp']}"):
                for j, (q_data, ans, feed) in enumerate(zip(session['questions'], session['answers'], session['feedbacks'])):
                    st.write(f"**Q{j+1}: {q_data['question']}** ({q_data['category']})")
                    st.write("*Your answer:*")
                    st.write(ans if ans else "*No answer recorded*")
                    if feed:
                        st.write("*Feedback:*")
                        st.write(feed)
                    st.divider()


# The interview screen
else:
    # Introduction phase
    if st.session_state.interview_stage == "introduction":
        try:
            # Create personalized introduction
            interviewee_name = st.session_state.interviewer_name or "candidate"
            job_role = st.session_state.selected_job_field
            
            intro_text = f"Hi {interviewee_name}, welcome to your interview practice for a {job_role} role. I'll be asking you a series of questions."
            
            # Generate audio for the introduction
            if st.session_state.use_voice:
                audio_fp = text_to_speech(intro_text)
                autoplay_audio(audio_fp)
            
            # Display the introduction text
            st.info(intro_text)
            
            # Show a "Continue" button to proceed to the first question
            if st.button("Continue to First Question", type="primary"):
                # Change the stage to question mode
                st.session_state.current_question_idx = 0
                st.session_state.interview_stage = "question"
                # Ensure question will be spoken
                st.session_state.question_spoken = False
                st.rerun()
                
        except Exception as e:
            st.error(f"Error in introduction: {str(e)}")
            if st.button("Continue to Questions"):
                st.session_state.interview_stage = "question"
                st.rerun()
    
    # Question phase
    elif st.session_state.interview_stage == "question":
        current_q_data = st.session_state.questions[st.session_state.current_question_idx]
        current_category = current_q_data["category"]
        current_question = current_q_data["question"]

        st.subheader("Question:")
        st.header(current_question)

        # Ensure the question gets spoken once
        if not st.session_state.question_spoken and st.session_state.use_voice:
            try:
                # Construct spoken question
                conversation_prefix = np.random.choice([
                   
                    
                    "For this question, ",
                   
                ])
                spoken_question = f"{conversation_prefix}{current_question}"

                # Generate TTS audio
                audio_fp = text_to_speech(spoken_question)

                if audio_fp:
                    autoplay_audio(audio_fp)
                    time.sleep(1)  # Ensure Streamlit renders before setting question_spoken
                    st.session_state.question_spoken = True
                else:
                    st.error("TTS audio was not generated correctly.")

            except Exception as e:
                st.error(f"Error playing question audio: {str(e)}")
                st.session_state.question_spoken = True  # Avoid re-triggering endlessly

        # Provide response input options
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.session_state.transcription:
                st.write("**Your transcribed answer:**")
                st.write(st.session_state.transcription)

                edited_answer = st.text_area(
                    "Edit your answer if needed:",
                    value=st.session_state.transcription,
                    height=150
                )

                if st.button("Save Answer & Continue"):
                    st.session_state.answers[st.session_state.current_question_idx] = edited_answer
                    with st.spinner("Generating feedback..."):
                        feedback = get_answer_feedback(current_question, edited_answer)
                        st.session_state.feedbacks[st.session_state.current_question_idx] = feedback

                    st.session_state.current_question_idx += 1
                    st.session_state.transcription = ""
                    st.session_state.audio_data = None
                    st.session_state.question_spoken = False

                    if st.session_state.current_question_idx >= len(st.session_state.questions):
                        st.session_state.interview_complete = True

                    st.rerun()

            else:
                audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=16000)

                if audio_bytes:
                    st.session_state.audio_data = audio_bytes
                    st.write("Processing your audio...")

                    with st.spinner("Transcribing..."):
                        transcript = transcribe_audio(audio_bytes)
                        st.session_state.transcription = transcript
                        st.rerun()

                st.write("Or type your answer:")
                text_answer = st.text_area("", height=150)

                if st.button("Submit Text Answer"):
                    if text_answer.strip():
                        st.session_state.transcription = text_answer
                        st.rerun()
                    else:
                        st.error("Please provide an answer before submitting.")

        if st.session_state.show_feedback:
            st.subheader("AI Feedback on Your Answer")
            
            with st.spinner("Generating feedback..."):
                if not st.session_state.feedbacks[st.session_state.current_question_idx]:
                    feedback = get_answer_feedback(current_question, st.session_state.transcription)
                    st.session_state.feedbacks[st.session_state.current_question_idx] = feedback
                else:
                    feedback = st.session_state.feedbacks[st.session_state.current_question_idx]
                
                st.write(feedback)
                
                if st.button("Continue to Next Question"):
                    st.session_state.answers[st.session_state.current_question_idx] = st.session_state.transcription
                    
                    st.session_state.current_question_idx += 1
                    st.session_state.transcription = ""
                    st.session_state.audio_data = None
                    st.session_state.show_feedback = False
                    st.session_state.question_spoken = False
                    
                    if st.session_state.current_question_idx >= len(st.session_state.questions):
                        st.session_state.interview_complete = True
                    
                    st.rerun()
