from .Static import Static
from .LMEO import LMEO

from dotenv import load_dotenv
import os

load_dotenv()
personal_api_key = os.environ.get('MY_API_KEY')
#anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')