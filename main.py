import asyncio
import streamlit as st
from utils.api_clients import (
    fetch_transcript, enhance_text, text_to_speech,
    get_available_voices
)
from utils.audio_processor import combine_audio_chunks, chunk_text
from utils.cache_manager import CacheManager
from config import MAX_CHUNK_LENGTH, CONCURRENT_TASKS

# Initialize cache manager
cache_manager = CacheManager()

def init_session_state():
    """Initialize session state variables"""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'progress' not in st.session_state:
        st.session_state.progress = 0

async def process_chunks(chunks: list, voice_id: str) -> bytes:
    """Process text chunks concurrently"""
    sem = asyncio.Semaphore(CONCURRENT_TASKS)
    audio_chunks = []
    progress_bar = st.progress(0)
    
    async def process_chunk(chunk: str) -> bytes:
        async with sem:
            enhanced = await enhance_text(chunk)
            return await text_to_speech(enhanced, voice_id)

    tasks = [process_chunk(chunk) for chunk in chunks]
    total_chunks = len(tasks)

    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        chunk_audio = await task
        if chunk_audio:
            audio_chunks.append(chunk_audio)
        progress = (i / total_chunks) * 100
        progress_bar.progress(int(progress))
        st.session_state.progress = progress

    return combine_audio_chunks(audio_chunks)

async def main():
    st.set_page_config(
        page_title="YouTube to Speech",
        page_icon="🎤",
        layout="wide"
    )

    init_session_state()

    st.title("YouTube Transcript to Speech Converter")
    st.markdown("Convert YouTube video transcripts to natural speech")

    # Input section
    col1, col2 = st.columns([2, 1])
    with col1:
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )

    # Voice selection
    voices = await get_available_voices()
    voice_options = {voice["name"]: voice["id"] for voice in voices}
    
    with col2:
        selected_voice = st.selectbox(
            "Select Voice",
            options=list(voice_options.keys())
        )

    if st.button("Convert to Speech", disabled=st.session_state.processing):
        if not video_url:
            st.error("Please enter a YouTube video URL")
            return

        st.session_state.processing = True
        try:
            with st.spinner("Processing video transcript..."):
                # Fetch and process transcript
                transcript = await fetch_transcript(video_url)
                text = " ".join(item["text"] for item in transcript)
                chunks = chunk_text(text, MAX_CHUNK_LENGTH)
                
                # Process chunks and generate audio
                voice_id = voice_options[selected_voice]
                audio_data = await process_chunks(chunks, voice_id)

                # Display audio player and download button
                st.audio(audio_data, format="audio/mp3")
                st.download_button(
                    label="Download Audio",
                    data=audio_data,
                    file_name="transcript_audio.mp3",
                    mime="audio/mp3"
                )

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
        finally:
            st.session_state.processing = False
            st.session_state.progress = 0

if __name__ == "__main__":
    asyncio.run(main())
