import os

from dotenv import load_dotenv

from src.utils import load_config

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATASETS_ACCESS_TOKEN = os.environ["DATASETS_ACCESS_TOKEN"]
PERSONAL_HF_TOKEN = os.environ["PERSONAL_HF_TOKEN"]
OPENAI_KEY = os.environ["OPENAI_KEY"]
WANDB_KEY = os.environ["WANDB_API_KEY"]
config = load_config("config.yaml")

os.environ["WANDB_PROJECT"] = str(config["run_config"]["WANDB_PROJECT"])
os.environ["WANDB_LOG_MODEL"] = str(config["run_config"]["WANDB_LOG_MODEL"])
