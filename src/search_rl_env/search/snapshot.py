"""Потоковая сборка и публикация неизменяемого SQLite-снимка."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from dataclasses import asdict
from collections.abc import Iterable

from .models import Document, InvalidRequest, SnapshotConflict, StorageError

SCHEMA_VERSION = 1
INDEX_ALGORITHM = "fts5-unicode61-document-bm25-v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def connect_readonly(path):
    # mode=ro не создаёт отсутствующий файл; опубликованный файл не изменяется.
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def load_manifest(db):
    manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["index_algorithm"] != INDEX_ALGORITHM:
        raise StorageError("Несовместимая версия снимка.")
    if manifest["sqlite_version"] != sqlite3.sqlite_version:
        raise StorageError("Версия SQLite отличается от версии сборки; пересоберите индекс.")
    return manifest


def build_snapshot(documents: Iterable[Document], path: str | Path, *, corpus_id: str,
                   corpus_revision: str | None = None, chunk_chars: int = 2000,
                   max_document_chars: int = 1_000_000) -> dict:
    """Собрать новый файл; существующий снимок никогда не перезаписывается.

    Повторная сборка идентичного снимка по тому же пути возвращает манифест.
    Вход — iterable Document, без вопросов и эталонной разметки.
    """
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise InvalidRequest("Нужен непустой corpus_id.")
    if corpus_revision is not None and (not isinstance(corpus_revision, str) or not corpus_revision.strip()):
        raise InvalidRequest("Ревизия должна быть непустой строкой.")
    for value in (chunk_chars, max_document_chars):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidRequest("Размеры документа и фрагмента должны быть положительными целыми.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".search-build-", suffix=".sqlite", dir=path.parent)
    os.close(fd)
    db = None
    try:
        db = sqlite3.connect(temporary)
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents(
                doc_id TEXT NOT NULL UNIQUE, text TEXT NOT NULL,
                title TEXT NOT NULL, language TEXT NOT NULL
            );
            CREATE TABLE chunks(
                chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                start_char INTEGER NOT NULL, end_char INTEGER NOT NULL
            );
            CREATE INDEX chunks_document ON chunks(doc_id, start_char);
            CREATE VIRTUAL TABLE document_fts USING fts5(
                title, text, content='documents', content_rowid='rowid', tokenize='unicode61'
            );
        """)
        for document in documents:
            if not isinstance(document, Document):
                raise InvalidRequest("Ожидался Document.")
            values = asdict(document)
            if any(not isinstance(value, str) for value in values.values()):
                raise InvalidRequest("Поля документа должны быть строками.")
            if not document.doc_id.strip() or not document.text.strip() or not document.language.strip():
                raise InvalidRequest("Id, текст и язык документа не могут быть пустыми.")
            if any("\x00" in value for value in values.values()):
                raise InvalidRequest("NUL в документе не поддерживается.")
            try:
                for value in values.values():
                    value.encode("utf-8")
            except UnicodeEncodeError:
                raise InvalidRequest("Поля документа должны быть корректным Unicode-текстом.") from None
            if len(document.text) > max_document_chars or len(document.title) > max_document_chars:
                raise InvalidRequest("Превышен размер документа.")
            try:
                db.execute("INSERT INTO documents VALUES (?, ?, ?, ?)",
                           (document.doc_id, document.text, document.title, document.language))
            except sqlite3.IntegrityError:
                raise InvalidRequest("Повторный doc_id во входном корпусе.") from None
            text_hash = fingerprint(document.text)
            for start in range(0, len(document.text), chunk_chars):
                end = min(start + chunk_chars, len(document.text))
                chunk_id = fingerprint([document.doc_id, text_hash, "chars-v1", chunk_chars, start, end])
                db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", (chunk_id, document.doc_id, start, end))

        digest = hashlib.sha256()
        count = 0
        for row in db.execute("SELECT doc_id, text, title, language FROM documents ORDER BY doc_id"):
            digest.update((canonical(row) + "\n").encode("utf-8"))
            count += 1
        if not count:
            raise InvalidRequest("Пустой корпус не поддерживается.")
        content_hash = digest.hexdigest()
        manifest = dict(schema_version=SCHEMA_VERSION, corpus_id=corpus_id,
                        corpus_revision=corpus_revision or content_hash, content_hash=content_hash,
                        chunk_chars=chunk_chars, chunk_algorithm="chars-v1", documents=count,
                        index_algorithm=INDEX_ALGORITHM, sqlite_version=sqlite3.sqlite_version)
        manifest["index_revision"] = fingerprint(manifest)
        db.execute("INSERT INTO document_fts(document_fts) VALUES ('rebuild')")
        db.execute("INSERT INTO document_fts(document_fts, rank) VALUES ('integrity-check', 1)")
        db.execute("INSERT INTO metadata VALUES ('manifest', ?)", (canonical(manifest),))
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise StorageError("Индекс не прошёл проверку целостности.")
        db.close()
        db = None
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        try:
            # link публикует готовый файл атомарно и не заменяет чужой результат.
            os.link(temporary, path)
        except FileExistsError:
            existing = connect_readonly(path)
            try:
                same = load_manifest(existing) == manifest
            finally:
                existing.close()
            if not same:
                raise SnapshotConflict("По этому пути уже опубликован другой снимок.") from None
        return manifest
    except (sqlite3.Error, OSError, UnicodeError):
        raise StorageError("Не удалось собрать или опубликовать снимок.") from None
    finally:
        if db is not None:
            db.close()
        os.unlink(temporary)
