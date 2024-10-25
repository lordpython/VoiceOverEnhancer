import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

# Constants
MAX_CHUNK_LENGTH = 500
CONCURRENT_TASKS = 5
CACHE_TTL = 86400  # 24 hours
