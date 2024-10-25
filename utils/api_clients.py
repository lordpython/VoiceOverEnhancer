import logging
import re
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
from openai import OpenAI
from config import OPENAI_API_KEY, ELEVENLABS_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

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
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": '''
                    Enhance and adjust the extracted text within the time frame and convert it to text used in voiceover videos:
                    1. Make it suitable for voiceover
                    2. Fix grammar and punctuation
                    3. Maintain natural flow
                    4. Ensure it will be interesting for the listener
                    5. Remove redundant words and repetitions
                    6. Add appropriate pauses with commas and periods
                    7. Format numbers and abbreviations for speech
                '''},
                {"role": "user", "content": text}
            ]
        )
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        return text
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return text

async def text_to_speech(
    text: str, 
    voice_id: str, 
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    use_speaker_boost: bool = True
) -> Optional[bytes]:
    """Convert text to speech using ElevenLabs with voice settings"""
    try:
        voice = Voice(
            voice_id=voice_id,
            settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=use_speaker_boost
            )
        )
        
        audio = elevenlabs_client.generate(
            text=text,
            voice=voice,
            model="eleven_turbo_v2"
        )
        # Convert iterator to bytes
        audio_bytes = b"".join(list(audio))
        return audio_bytes
    except Exception as e:
        logger.error(f"ElevenLabs API error: {e}")
        return None

async def get_available_voices() -> List[Dict[str, str]]:
    """Fetch available voices from ElevenLabs"""
    try:
        response = elevenlabs_client.voices.get_all()
        voices = []
        for voice in response.voices:
            voice_dict = {
                "name": voice.name,
                "id": voice.voice_id
            }
            voices.append(voice_dict)
        return voices
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return []
