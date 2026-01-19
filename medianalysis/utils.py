import logging
import json

def setup_logging(level: str = 'INFO', log_file: str = ''):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level = getattr(logging, level.upper()),
        format = log_format,
        handlers = handlers
    )

def save_json(data, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
