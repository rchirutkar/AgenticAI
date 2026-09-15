import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM
from dotenv import load_dotenv
from google import genai  # 👈 Import the official Google GenAI library
import io

# Load environment keys from .env file
load_dotenv()

# Streamlit UI Configuration
st.set_page_config(page_title="AI Kids Storyteller", page_icon="✨", layout="wide")
st.title("✨ Expressive AI Kids Storyteller")
st.write("Generate stories with Gemini 3.5 Flash-Lite and narrate them with ultra-realistic Gemini TTS!")

# Sidebar Configuration Controls
with st.sidebar:
    st.header("🎨 Story Settings")
    theme = st.text_input("Story Theme / Topic", value="A brave little turtle learning to fly")
    age_group = st.slider("Target Age Group", min_value=3, max_value=12, value=6)
    
    st.header("🎭 Tone & Emotion Direction")
    healthy_tone = st.selectbox(
        "Core Wholesome Value", 
        ["Kindness & Empathy", "Resilience & Growth", "Curiosity & Science", "Honesty & Friendship"]
    )
    
    # 🎭 Director Prompt Settings to make the voice sound non-robotic
    vocal_style = st.selectbox(
        "Voice Delivery Style", 
        ["Warm & Cozy Bedtime Story", "Exciting & Energetic Adventure", "Curious & Whispering Mystery"]
    )
    
    st.header("🎙️ Voice Settings")
    enable_voice = st.checkbox("Enable Ultra-Realistic Narration", value=True)

# Define CrewAI Generation Logic
def generate_children_story(theme, age, tone):
    
    custom_llm = LLM(model="gemini/gemini-3.5-flash-lite", temperature=1.0)

    plotter = Agent(
        role="Children's Story Plotter",
        goal="Design a safe, engaging, and structurally sound story outline for kids.",
        backstory="An expert child psychologist who specializes in positive reinforcement and gentle conflicts.",
        verbose=True,
        llm=custom_llm
    )

    author = Agent(
        role="Children's Book Author",
        goal=f"Write beautiful storytelling prose between 300 and 500 words emphasizing {tone}.",
        backstory="An award-winning children's author who weaves magic with short, human-like sentences.",
        verbose=True,
        llm=custom_llm
    )

    task_plot = Task(
        description=f"Create a short 3-chapter plot outline based on the theme: '{theme}' for a {age}-year-old.",
        expected_output="An outline containing a gentle conflict and a positive resolution.",
        agent=plotter
    )

    task_write = Task(
        description=(
            f"Write the full story based on the outline. Total word count MUST be between 300 and 500 words. "
            f"Infuse a healthy, encouraging tone focused on {tone}. Crucially, insert explicit '[pause]' markers "
            "at the end of key sentences or paragraphs to indicate narrator pacing adjustments."
        ),
        expected_output="A beautiful 300-500 word children's story divided into chapters with text and '[pause]' elements.",
        agent=author
    )

    story_crew = Crew(agents=[plotter, author], tasks=[task_plot, task_write], process=Process.sequential)
    return story_crew.kickoff()

# Main Application Trigger
if st.button("🚀 Generate Story Magic"):
    with st.spinner("🔮 The AI Storytellers are writing and rehearsing the performance..."):
        
        # 1. Run CrewAI to generate the story text
        raw_output = generate_children_story(theme, age_group, healthy_tone)
        story_text = f"{raw_output}"
        
        st.success("📖 Creative Writing & Recording Complete!")
        
        # 2. Process text for UI display
        clean_display_text = story_text.replace("[pause]", "\n\n⏱️ *[Pause for breath]* \n\n")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📚 Your Wholesome Adventure")
            st.write(clean_display_text)
            
        with col2:
            st.markdown("### 🎙️ Audio Production Room")
            st.image("https://unsplash.com", 
                     caption="Story Backdrop")
            
            if enable_voice:
                try:
                    # Clean punctuation formatting markers for standard audio reading
                    audio_reading_text = story_text.replace("[pause]", " ... ")
                    
                    # Directing guidance to adjust style parameters
                    director_guidance = (
                        f"Please act as a voice narrator. Deliver this story with a {vocal_style.lower()} tone. "
                        f"Speak slowly and warmly for a child who is {age_group} years old. "
                    )
                    
                    full_speech_prompt = f"{director_guidance}\n\nStory Script:\n{audio_reading_text}"
                    
                    # Initialize the direct SDK client 
                    client = genai.Client()
                    
                    # Call standard generation with explicit audio modality requested
                    response = client.models.generate_content(
                        model='gemini-2.5-flash-preview-tts',  # 👈 Updated Preview identifier string
                        contents=full_speech_prompt,
                        config={
                            "response_modalities": ["AUDIO"]    # 👈 Explicitly request audio waveform media
                        }
                    )
                    
                    # Safely search and extract the raw binary stream data from payload parts
                    audio_bytes = None
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            # Extract inline media data if present
                            if hasattr(part, 'inline_data') and part.inline_data:
                                audio_bytes = part.inline_data.data
                                break
                    
                    if audio_bytes:
                        st.info(f"🎧 Narration Style: **{vocal_style}**")
                        # Present the fully expressive, human-like voice track
                        st.audio(audio_bytes, format="audio/wav")
                    else:
                        st.warning("The voice model processed your prompt but did not return a valid audio track part.")
                        
                except Exception as e:
                    st.error(f"Voice generation failed: {e}")

