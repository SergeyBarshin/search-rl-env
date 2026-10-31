"""Чтение ключа API из локального окружения или выбранного файла."""

import os
from pathlib import Path


def load_api_key(*, key_file: Path | None = None) -> str:
    key = os.getenv("MODEL_API_KEY", "").strip()
    if key:
        return key
    configured_file = os.getenv("MODEL_API_KEY_FILE", "").strip()
    if key_file is None and not configured_file:
        raise ValueError("Нужен MODEL_API_KEY или MODEL_API_KEY_FILE.")
    key_file = key_file if key_file is not None else Path(configured_file).expanduser()
    try:
        key = key_file.read_text(encoding="utf-8").strip()
    except OSError:
        raise ValueError("Не удалось прочитать файл ключа API.") from None
    if not key:
        raise ValueError("Файл ключа API пуст.")
    return key
