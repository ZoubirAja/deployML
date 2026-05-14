import os
from dotenv import load_dotenv

APP_ENV = os.getenv("APP_ENV", "dev")

env_file = f".env.test"
if os.path.exists(env_file):
    load_dotenv(env_file)
else:
    load_dotenv(".env.test")  # fallback

DEBUG = os.getenv("DEBUG", "false").lower() == "true"