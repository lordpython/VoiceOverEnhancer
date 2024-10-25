import io
from typing import List
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)

def combine_audio_chunks(chunks: List[bytes]) -> bytes:
    """Combine multiple audio chunks into a single audio file"""
    try:
        combined = AudioSegment.empty()
        for chunk in chunks:
            audio_segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
            combined += audio_segment
        
        # Export as MP3
        buffer = io.BytesIO()
        combined.export(buffer, format="mp3")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error combining audio chunks: {e}")
        raise ValueError("Failed to combine audio chunks")

def chunk_text(text: str, max_length: int) -> List[str]:
    """Split text into processable chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + 1 <= max_length:
            current_chunk.append(word)
            current_length += word_length + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
