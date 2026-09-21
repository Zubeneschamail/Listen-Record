"""Explicit one-time preparation for source users; no customer data is accessed."""
from app_paths import MODELS
from knowledge_embedding import prepare_model

if __name__ == '__main__':
    prepare_model(MODELS / 'bge-small-zh-v1.5')
