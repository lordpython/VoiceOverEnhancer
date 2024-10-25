import logging
import re
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from elevenlabs import generate, set_api_key, Voices
import openai
from config import OPENAI_API_KEY, ELEVENLABS_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure API keys
openai.api_key = OPENAI_API_KEY
set_api_key(ELEVENLABS_API_KEY)

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL"""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL format")
    return match.group(1)

async def fetch_transcript(video_url: str) -> List[Dict[str, Any]]:
    """Fetch transcript from YouTube"""
    try:
        video_id = extract_video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        raise ValueError(f"Transcript error: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching transcript: {e}")
        raise ValueError("Failed to fetch transcript")

async def enhance_text(text: str) -> str:
    """Enhance text using OpenAI"""
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Enhance this transcript text while maintaining its meaning:"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return text

async def text_to_speech(text: str, voice_id: str) -> Optional[bytes]:
    """Convert text to speech using ElevenLabs"""
    try:
        audio = await generate.async_generate(
            text=text,
            voice=voice_id,
            api_key=ELEVENLABS_API_KEY
        )
        return audio
    except Exception as e:
        logger.error(f"ElevenLabs API error: {e}")
        return None

async def get_available_voices() -> List[Dict[str, str]]:
    """Fetch available voices from ElevenLabs"""
    try:
        voices = await Voices.async_get_all()
        return [{"name": voice.name, "id": voice.voice_id} for voice in voices]
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return []
