import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit_TTS import text_to_audio, auto_play  # Import voice engines
from gtts import gTTS 
from google import genai # Import the official Google GenAI library
import io



load_dotenv()  # This loads the keys from your .env file into the system memory


# 1. Streamlit UI Configuration
st.set_page_config(page_title="AI Kids Storyteller", page_icon="✨", layout="wide")
st.title("✨ AI Children's Storyteller & Voice Narrator")
st.write("Create wholesome, engaging stories with custom themes and pacing adjustments and hear them narrated aloud!.")

# Sidebar Configuration Controls
with st.sidebar:
    st.header("🎨 Story Settings")
    theme = st.text_input("Story Theme / Topic", value="A brave little turtle learning to fly")
    age_group = st.slider("Target Age Group", min_value=3, max_value=12, value=6)
    
    st.header("🎭 Tone & Audio Control")
    healthy_tone = st.selectbox(
        "Core Wholesome Value", 
        ["Kindness & Empathy", "Resilience & Growth", "Curiosity & Science", "Honesty & Friendship"]
    )

    st.header("🎙️ Voice Settings")
    enable_voice = st.checkbox("Enable Audio Narration", value=True)

    pause_duration = st.slider("Narration Pause Length (seconds)", 0.5, 3.0, 1.5, step=0.5)

# 2. Define CrewAI Agents and Tasks
def generate_children_story(theme, age, tone):
    # Setup LLM provider
    #llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Switch to Gemini 1.5 Pro
    # llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)

    # Agent 1: The Wholesome Story Outliner
    # plotter = Agent(
    #     role="Children's Story Plotter",
    #     goal="Design a safe, engaging, and structurally sound story outline for kids.",
    #     backstory="An expert child psychologist and creative outline writer who specializes in positive reinforcement.",
    #     verbose=True,
    #     llm=llm
    # )

    custom_llm = LLM(
        model="gemini/gemini-3.5-flash-lite", 
        temperature=0.7
    )

    # Agent 1: The Wholesome Story Outliner
    plotter = Agent(
        role="Children's Story Plotter",
        goal="Design a safe, engaging, and structurally sound story outline for kids.",
        backstory="An expert child psychologist who specializes in positive reinforcement.",
        verbose=True,
        llm=custom_llm # 
    )

    # Agent 2: The Creative Children's Author
    author = Agent(
        role="Children's Book Author",
        goal=f"Write a rich, captivating story between 300 and 500 words emphasizing {tone}.",
        backstory="An award-winning children's author who weaves magic with words, ensuring a gentle, uplifting tone.",
        verbose=True,
        llm=custom_llm
    )

    # Define the tasks for the agents
    task_plot = Task(
        description=f"Create a 3-chapter plot outline based on the theme: '{theme}' for a {age}-year-old.",
        expected_output="An outline containing a gentle conflict and a positive resolution.",
        agent=plotter
    )

    task_write = Task(
        description=(
            f"Write the full story based on the outline. The total word count MUST be between 300 and 500 words. "
            f"Infuse a healthy, encouraging tone focused on {tone}. Crucially, insert explicit '[pause]' markers "
            "at the end of dramatic or emotional sentences to indicate audio narrative pacing adjustments."
        ),
        expected_output="A beautiful 300-500 word children's story divided into 3 chapters with text and '[pause]' elements.",
        agent=author
    )

    # Form the Crew
    story_crew = Crew(
        agents=[plotter, author],
        tasks=[task_plot, task_write],
        process=Process.sequential
    )

    return story_crew.kickoff()

# 3. Main Application Trigger

# if st.button("🚀 Generate Story Magic"):
    with st.spinner("🔮 The AI Storytellers are brainstorming and recording..."):
        
        # 1. Run CrewAI to draft the story script
        raw_output = generate_children_story(theme, age_group, healthy_tone)
        story_text = f"{raw_output}"
        
        st.success("📖 Story and Voice Generation Complete!")

         # 2. Process Text for Visual Display (Hide ugly formatting tags if any)
        clean_display_text = story_text.replace("[pause]", "\n\n⏱️ *[Pause for breath]* \n\n")
        
        
         # 3. Split Layout for Presentation
        col1, col2 = st.columns([2, 1])
        
        # with col1:
        #     st.markdown("### 📚 Your Wholesome Adventure")
        #     text_story = f"{story_result}" 
        #     st.write(text_story)
            
        # with col2:
        #     st.markdown("### 🎙️ Production Elements")
        #     st.image("https://unsplash.com", 
        #              caption="Generated Story Backdrop Concept")

        with col1:
            st.markdown("### 📚 Your Wholesome Adventure")
            st.write(clean_display_text)
            
        with col2:
            st.markdown("### 🎙️ Audio Production Room")
            st.image("https://unsplash.com", 
                     caption="Generated Story Backdrop Concept")
            
            # 4. Generate Voice Data dynamically if checked
            if enable_voice:
                # Strip out formatting symbols so the voice module reads raw words smoothly
                audio_reading_text = story_text.replace("[pause]", " . . . ")
                
                # Convert prose words directly to readable waveform buffers
                audio_data = text_to_audio(audio_reading_text, language='en')
                
                if audio_data and 'bytes' in audio_data:
                    st.info("🎧 Listen to your story narration below:")
                    # Render the standard browser audio media control player bar on screen
                    st.audio(audio_data['bytes'], format="audio/wav")
                else:
                    st.warning("Could not render narration buffer.")


# --- Main Application Trigger ---
if st.button("🚀 Generate Story Magic"):
    with st.spinner("🔮 The AI Storytellers are brainstorming and recording..."):
        
        # 1. Run CrewAI
        raw_output = generate_children_story(theme, age_group, healthy_tone)
        story_text = f"{raw_output}"
        
        st.success("📖 Story Complete!")
        
        # 2. Process text for UI display
        clean_display_text = story_text.replace("[pause]", "\n\n⏱️ *[Pause for breath]* \n\n")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📚 Your Wholesome Adventure")
            st.write(clean_display_text)
            
        with col2:
            st.markdown("### 🎙️ Audio Production Room")
            st.image("https://unsplash.com", 
                     caption="Generated Story Backdrop Concept")
            
            if enable_voice:
                try:
                    # Clean the story text for a smooth voice read
                    audio_reading_text = story_text.replace("[pause]", " ... ")
                    
                    # 3. Create the audio object via standard gTTS
                    tts = gTTS(text=audio_reading_text, lang='en', slow=False)
                    
                    # 4. Stream bytes straight into an in-memory buffer 
                    # This avoids saving messy physical temp files in WSL!
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    
                    st.info("🎧 Listen to your story narration below:")
                    # Render the browser native player using the safe raw bytes
                    st.audio(audio_buffer.read(), format="audio/mp3")
                    
                except Exception as e:
                    st.error(f"Audio processing glitch: {e}")

