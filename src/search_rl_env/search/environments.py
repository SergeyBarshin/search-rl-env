"""Подготовка набора один раз и создание собственных SQLite/FAISS для эпизода."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .local import (FaissSearch, HashingEncoder, SQLiteSnapshot, digest_file,
                    fingerprint, joined_thread, require_standalone)
from .models import SearchProtocolError, StorageError


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    version: Literal[1]
    dataset_id: str = Field(min_length=1)
    sql_path: str = Field(min_length=1)
    sql_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sql_format: Literal["sqlite", "fixture_sql"] = "sqlite"
    db_id: str = Field(min_length=1)
    corpus_path: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_id: str = Field(min_length=1)
    dimensions: int = Field(default=128, ge=16, le=4096)


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    doc_id: str = Field(min_length=1, max_length=4096)
    text: str = Field(min_length=1, max_length=1_000_000)
    title: str = Field(default="", max_length=1_000_000)
    language: str = Field(default="und", min_length=1, max_length=64)

    @field_validator("doc_id", "text", "title", "language")
    @classmethod
    def valid_text(cls, value, info):
        if "\0" in value or info.field_name != "title" and not value.strip():
            raise ValueError("Пустая строка или NUL.")
        value.encode("utf-8", errors="strict")
        return value


@dataclass(frozen=True)
class DatasetData:
    spec: DatasetSpec
    sqlite_path: Path
    documents: tuple[dict, ...]
    manifest: dict


def prepare_dataset(manifest_path: Path, directory: Path) -> DatasetData:
    """fixture_sql — доверенный локальный build-вход, никогда не SQL агента."""
    manifest_path = Path(manifest_path).resolve()
    spec = DatasetSpec.model_validate_json(manifest_path.read_bytes())
    sql_source = (manifest_path.parent / spec.sql_path).resolve()
    corpus_source = (manifest_path.parent / spec.corpus_path).resolve()
    if digest_file(sql_source) != spec.sql_sha256 or digest_file(corpus_source) != spec.corpus_sha256:
        raise StorageError("Исходные данные не соответствуют SHA-256 манифеста.")
    documents, identifiers = [], set()
    with corpus_source.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            document = CorpusDocument.model_validate_json(line).model_dump()
            if document["doc_id"] in identifiers:
                raise ValueError("Повторный doc_id внутри корпуса.")
            identifiers.add(document["doc_id"])
            documents.append(document)
    if not documents:
        raise ValueError("Корпус пуст.")
    documents.sort(key=lambda item: item["doc_id"])
    directory.mkdir(parents=True, exist_ok=True)
    sqlite_path = directory / "source.sqlite"
    if spec.sql_format == "fixture_sql":
        if sql_source.stat().st_size > 1_000_000:
            raise ValueError("Синтетический SQL fixture превышает 1 МБ.")
        db = sqlite3.connect(sqlite_path)
        try:
            db.executescript(sql_source.read_text(encoding="utf-8"))
            db.commit()
        finally:
            db.close()
        # Build-скрипт не должен оставлять опубликованную БД в WAL-режиме.
        require_standalone(sqlite_path)
    else:
        require_standalone(sql_source)
        shutil.copyfile(sql_source, sqlite_path)
        if digest_file(sqlite_path) != spec.sql_sha256:
            raise StorageError("SQL-источник изменился во время копирования.")
    revision = digest_file(sqlite_path)
    probe = SQLiteSnapshot(sqlite_path, spec.db_id, revision)
    # Здесь нет активных workers; закрываем синхронно до публикации шаблона.
    probe.close()
    content_hash = fingerprint(documents)
    encoder = HashingEncoder(spec.dimensions)
    dataset_revision = fingerprint({"sql_source": spec.sql_sha256, "corpus_source": spec.corpus_sha256,
                                    "format": spec.sql_format, "encoder": encoder.manifest})
    metadata = {"dataset_id": spec.dataset_id, "dataset_revision": dataset_revision,
                "sql": {"db_id": spec.db_id, "db_revision": revision, "schema": probe.schema,
                        "source_sha256": spec.sql_sha256, "source_format": spec.sql_format,
                        "sqlite_version": sqlite3.sqlite_version, "policy_revision": "sqlite-authorizer-read-v1"},
                "search": {"corpus_id": spec.corpus_id, "corpus_revision": content_hash,
                           "documents": len(documents), "encoder": encoder.manifest,
                           "index": "IndexFlatIP", "score": "cosine", "snippet_chars": 500}}
    return DatasetData(spec, sqlite_path, tuple(documents), metadata)


class EpisodeEnvironment:
    def __init__(self, owner, identifier, directory, sql, search, manifest):
        self._owner, self.environment_id, self.directory = owner, identifier, directory
        self.sql, self.search, self._manifest = sql, search, deepcopy(manifest)
        self.closed = False

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def validate_task(self, task):
        scope = self._manifest["search"]
        if task.corpus_id != scope["corpus_id"] or task.corpus_revision not in (None, scope["corpus_revision"]):
            raise ValueError("Задача относится к другому корпусу или ревизии.")

    async def aclose(self):
        if self.closed:
            return
        # Factory/context manager могут закрывать одну среду одновременно.
        async with self._close_lock:
            if self.closed:
                return
            try:
                try:
                    await self.search.aclose()
                finally:
                    await self.sql.aclose()
            finally:
                try:
                    await joined_thread(shutil.rmtree, self.directory)
                finally:
                    self.closed = True
                    self._owner._active.pop(self.environment_id, None)


class EnvironmentFactory:
    """Async lifetime принадлежит runner; у каждого open своя БД и FAISS clone."""

    def __init__(self, manifests, *, sql_timeout_seconds=1.0, max_vm_steps=100_000, directory=None):
        self._manifests = [Path(path) for path in manifests]
        if not self._manifests:
            raise ValueError("Нужен хотя бы один набор.")
        self._parent = directory
        self._sql_timeout, self._max_vm_steps = sql_timeout_seconds, max_vm_steps
        self._active, self._datasets = {}, {}
        self._temporary = None
        self.closed = False
        self.peak_active = 0
        self.preparation_seconds = None

    @property
    def active_count(self):
        return len(self._active)

    @property
    def manifests(self):
        return {key: json.loads(json.dumps(value[0].manifest)) for key, value in self._datasets.items()}

    async def export(self, directory):
        """Артефакты подготовки для runner/MLflow; эталонов здесь нет."""
        if self.closed or self._temporary is None:
            raise SearchProtocolError("Фабрика закрыта или ещё не открыта.")
        directory = Path(directory)
        exported = {}
        for identifier, (data, index, _) in self._datasets.items():
            folder = directory / fingerprint(identifier)[:16]
            folder.mkdir(parents=True)
            await joined_thread(shutil.copyfile, data.sqlite_path, folder / "source.sqlite")
            await joined_thread(shutil.copyfile, data.sqlite_path.with_suffix(".faiss"), folder / "index.faiss")
            (folder / "corpus.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                       for row in data.documents), encoding="utf-8")
            metadata = json.loads(json.dumps(data.manifest))
            metadata["artifacts"] = {name: digest_file(folder / name)
                                     for name in ("source.sqlite", "index.faiss", "corpus.jsonl")}
            (folder / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            exported[identifier] = str((folder / "manifest.json").relative_to(directory))
        return exported

    async def __aenter__(self):
        if self.closed or self._temporary is not None:
            raise SearchProtocolError("Фабрика уже открыта или закрыта.")
        try:
            import faiss
        except ImportError:
            self.closed = True
            raise RuntimeError("Для сред установите search-rl-env[environments]; нужны настоящие FAISS и NumPy.") from None
        self._faiss = faiss
        self._temporary = tempfile.TemporaryDirectory(prefix="search-env-", dir=self._parent)
        started = perf_counter()
        try:
            for path in self._manifests:
                directory = Path(self._temporary.name) / uuid4().hex
                data = await joined_thread(prepare_dataset, path, directory)
                if data.spec.dataset_id in self._datasets:
                    raise ValueError("Повторный dataset_id.")
                data.manifest["sql"]["config"] = {"timeout_seconds": self._sql_timeout,
                    "max_vm_steps": self._max_vm_steps, "max_rows": 100, "max_columns": 100,
                    "max_result_bytes": 256 * 1024, "max_value_bytes": 1_000_000}
                encoder = HashingEncoder(data.spec.dimensions)
                vectors = await joined_thread(encoder.encode,
                    [document["title"] + "\n" + document["text"] for document in data.documents])
                index = faiss.IndexFlatIP(data.spec.dimensions)
                await joined_thread(index.add, vectors)
                index_path = data.sqlite_path.with_suffix(".faiss")
                await joined_thread(faiss.write_index, index, str(index_path))
                metadata = data.manifest["search"]
                metadata["faiss_version"] = version("faiss-cpu")
                metadata["numpy_version"] = version("numpy")
                metadata["index_content_sha256"] = digest_file(index_path)
                metadata["index_revision"] = fingerprint(metadata)
                metadata["search_revision"] = fingerprint({"index_revision": metadata["index_revision"],
                    "rank_policy": "score-desc-doc-id-asc-v1", "snippet_chars": 500, "max_results": 100})
                self._datasets[data.spec.dataset_id] = (data, index, encoder)
            self.preparation_seconds = perf_counter() - started
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        try:
            for environment in list(self._active.values()):
                await environment.aclose()
        finally:
            self._datasets.clear()
            if self._temporary is not None:
                await joined_thread(self._temporary.cleanup)

    @asynccontextmanager
    async def open(self, dataset_id):
        if self.closed or self._temporary is None:
            raise SearchProtocolError("Используйте фабрику через async with.")
        if dataset_id not in self._datasets:
            raise ValueError("Неизвестный dataset_id.")
        started = perf_counter()
        data, template, encoder = self._datasets[dataset_id]
        identifier = uuid4().hex
        directory = Path(self._temporary.name) / ("episode-" + identifier)
        directory.mkdir()
        sql = environment = None
        try:
            path = directory / "episode.sqlite"
            await joined_thread(shutil.copyfile, data.sqlite_path, path)
            sql = SQLiteSnapshot(path, data.spec.db_id, data.manifest["sql"]["db_revision"],
                                 timeout_seconds=self._sql_timeout, max_vm_steps=self._max_vm_steps)
            index = await joined_thread(self._faiss.clone_index, template)
            manifest = json.loads(json.dumps(data.manifest))
            manifest["environment_id"] = identifier
            manifest["creation_seconds"] = perf_counter() - started
            search = FaissSearch(index, data.documents, encoder, manifest["search"])
            environment = EpisodeEnvironment(self, identifier, directory, sql, search, manifest)
            environment._close_lock = asyncio.Lock()
            if self.closed:
                raise SearchProtocolError("Фабрика закрылась во время создания среды.")
            self._active[identifier] = environment
            self.peak_active = max(self.peak_active, len(self._active))
            yield environment
        finally:
            if environment is not None:
                await environment.aclose()
            else:
                if sql is not None:
                    await sql.aclose()
                await joined_thread(shutil.rmtree, directory)
