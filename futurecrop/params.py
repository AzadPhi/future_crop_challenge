from dotenv import load_dotenv
import os

load_dotenv()

CROP = os.environ.get('CROP')
MODE = os.environ.get('MODE')

# paths for data and submission directories and model trained
DATA_DIRECTORY = os.environ.get('DATA_DIRECTORY')
PATH_TO_MODEL = os.environ.get('PATH_TO_MODEL')
SUBMISSION_DIRECTORY = os.environ.get('SUBMISSION_DIRECTORY')
