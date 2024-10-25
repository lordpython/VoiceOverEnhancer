import asyncio
import streamlit as st
from utils.api_clients import (fetch_transcript, enhance_text, text_to_speech,
                               get_available_voices)
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


async def process_chunks(chunks: list, voice_id: str,
                         voice_settings: dict) -> bytes:
    """Process text chunks concurrently"""
    sem = asyncio.Semaphore(CONCURRENT_TASKS)
    audio_chunks = []
    progress_bar = st.progress(0)

    async def process_chunk(chunk: str) -> bytes:
        async with sem:
            enhanced = await enhance_text(chunk)
            audio = await text_to_speech(
                enhanced,
                voice_id,
                stability=voice_settings['stability'],
                similarity_boost=voice_settings['similarity_boost'],
                style=voice_settings['style'],
                use_speaker_boost=voice_settings['speaker_boost'])
            return audio if audio else b""

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
    st.set_page_config(page_title="محول نصوص يوتيوب إلى كلام",
                       page_icon="🎤",
                       layout="wide")

    # Add CSS for RTL support
    st.markdown("""
        <style>
        .stTextInput, .stSelectbox, .stSlider, .stCheckbox {
            direction: rtl;
            text-align: right;
        }
        .css-1inwz65 {
            direction: rtl;
            text-align: right;
        }
        </style>
    """,
                unsafe_allow_html=True)

    init_session_state()

    st.title("تحويل النص الى تعليق صوتي")
    st.markdown("تحويل نصوص فديوات يوتيوب إلى تعليق صوتي.")

    # Input section
    col1, col2 = st.columns([2, 1])
    with col1:
        video_url = st.text_input(
            "رابط فيديو يوتيوب",
            placeholder="https://www.youtube.com/watch?v=...")

    # Voice selection and settings
    voices = await get_available_voices()
    if not voices:
        st.error("فشلت العملية يا انه الفديو بطاط او تدق على يوسف ")
        return

    voice_options = {voice["name"]: voice["id"] for voice in voices}

    with col2:
        selected_voice = st.selectbox("شوفلك واحد صوته طيب",
                                      options=list(voice_options.keys()))

    # Voice configuration section
    st.subheader("إعدادات الصوت")
    col3, col4, col5 = st.columns(3)

    with col3:
        stability = st.slider(
            "الثبات",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="تعلييه يثبت الصوت تقلل مادري عاد شيصير بالضبط")

        style = st.slider(
            "النمط",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            help="القيم الأعلى تعزز النمط والمشاعر ولكن قد تؤثر على التماسك")

    with col4:
        similarity_boost = st.slider(
            "تعزيز التشابه",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            help=
            "اذا زدت بهذا الصةت راح يصير نفس الصوت الصجي اذا ماكان في صوت صجي بالاساس يمكن يطلعلك اللي يقول ولك الوووه"
        )

    with col5:
        speaker_boost = st.checkbox(
            "تعزيز الصوت",
            value=True,
            help="تحسين وضوح الصوت وتقليل ضوضاء الخلفية")

    voice_settings = {
        'stability': stability,
        'similarity_boost': similarity_boost,
        'style': style,
        'speaker_boost': speaker_boost
    }

    if st.button("تحويل إلى كلام", disabled=st.session_state.processing):
        if not video_url:
            st.error("الرجاء إدخال رابط فيديو يوتيوب")
            return

        st.session_state.processing = True
        try:
            with st.spinner("جارٍ معالجة نص الفيديو..."):
                # Fetch and process transcript
                transcript = await fetch_transcript(video_url)
                text = " ".join(item["text"] for item in transcript)
                chunks = chunk_text(text, MAX_CHUNK_LENGTH)

                # Process chunks and generate audio
                voice_id = voice_options[selected_voice]
                audio_data = await process_chunks(chunks, voice_id,
                                                  voice_settings)

                # Display audio player and download button
                st.audio(audio_data, format="audio/mp3")
                st.download_button(label="تحميل الصوت",
                                   data=audio_data,
                                   file_name="transcript_audio.mp3",
                                   mime="audio/mp3")

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"حدث خطأ غير متوقع: {str(e)}")
        finally:
            st.session_state.processing = False
            st.session_state.progress = 0


if __name__ == "__main__":
    asyncio.run(main())
