"""Python HTTP-сервис SQLite FTS5/BM25 и read-only SQL."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import re
import sqlite3

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field

from search_rl_env.contracts import SqlQueryError
from .environments import CorpusDocument
from .local import SQLiteSnapshot, digest_file, fingerprint, joined_thread, require_standalone
from .models import Manifest, SearchError, StorageError, UnknownId


CONFIG = dict(max_results=100, snippet_chars=500, max_read_chars=8000,
              max_query_chars=2000, max_query_terms=32, workers=16,
              max_pending=128, timeout_nanoseconds=5_000_000_000)
ALGORITHM = 'python-fts5-unicode61-document-bm25-v1'


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    corpus_id: str
    corpus_revision: str | None = None


class SearchRequest(Request):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=100)


class ReadRequest(Request):
    doc_id: str | None = None
    chunk_id: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8000, ge=1, le=8000)


class ChunksRequest(Request):
    doc_id: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)


class SQLRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    db_id: str
    db_revision: str
    query: str


def build_snapshot(source, output, corpus_id, revision=None, chunk_chars=2000):
    if not corpus_id.strip() or not 1 <= chunk_chars <= 1_000_000:
        raise ValueError('Недопустимые параметры корпуса.')
    documents = [CorpusDocument.model_validate_json(line).model_dump()
                 for line in Path(source).read_text(encoding='utf-8').splitlines() if line.strip()]
    documents.sort(key=lambda row: row['doc_id'])
    if not documents or len({row['doc_id'] for row in documents}) != len(documents):
        raise ValueError('Пустой корпус или повторный doc_id.')
    manifest = dict(schema_version=2, corpus_id=corpus_id,
        corpus_revision=revision or fingerprint(documents), content_hash=fingerprint(documents),
        chunk_chars=chunk_chars, chunk_algorithm='chars-v1', documents=len(documents),
        index_algorithm=ALGORITHM, sqlite_version=sqlite3.sqlite_version, driver_version='stdlib-sqlite3')
    manifest['index_revision'] = fingerprint(manifest)
    output = Path(output)
    if output.exists():
        existing = SearchStore(output)
        if any(existing.manifest[key] != value for key, value in manifest.items()):
            raise ValueError('Путь занят другим снимком.')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile, os
    fd, name = tempfile.mkstemp(prefix='.search-build-', dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    db = sqlite3.connect(temporary)
    try:
        db.executescript('''CREATE TABLE metadata(value TEXT NOT NULL);
        CREATE TABLE documents(doc_id TEXT UNIQUE NOT NULL, text TEXT NOT NULL, title TEXT NOT NULL, language TEXT NOT NULL);
        CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT, start_char INTEGER, end_char INTEGER);
        CREATE INDEX chunks_document ON chunks(doc_id,start_char);
        CREATE VIRTUAL TABLE document_fts USING fts5(title,text,content='documents',content_rowid='rowid',tokenize='unicode61');''')
        for document in documents:
            db.execute('INSERT INTO documents VALUES (?,?,?,?)', [document[k] for k in ('doc_id','text','title','language')])
            text_hash = fingerprint(document['text'])
            for start in range(0, len(document['text']), chunk_chars):
                end = min(start + chunk_chars, len(document['text']))
                identifier = fingerprint([document['doc_id'], text_hash, chunk_chars, start, end])
                db.execute('INSERT INTO chunks VALUES (?,?,?,?)', (identifier, document['doc_id'], start, end))
        db.execute("INSERT INTO document_fts(document_fts) VALUES ('rebuild')")
        db.execute('INSERT INTO metadata VALUES (?)', (json.dumps(manifest),))
        db.commit()
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        db.close()
        try:
            os.link(temporary, output)
        except FileExistsError:
            existing = SearchStore(output)
            if any(existing.manifest[k] != value for k, value in manifest.items()):
                raise ValueError('Путь занят другим снимком.')
    finally:
        db.close()
        temporary.unlink(missing_ok=True)
    return manifest


class SearchStore:
    def __init__(self, path, *, sealed=False):
        self.path = Path(path).resolve()
        self._sealed = sealed
        require_standalone(self.path)
        self._snapshot_stat = self._stat_signature()
        with self.connection() as db:
            manifest = json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])
        self.manifest = Manifest.model_validate({**manifest, 'api_version': '1',
            'search_revision': fingerprint([manifest, CONFIG]), 'config': CONFIG}).model_dump()
        self.revision = digest_file(self.path)
        if self._stat_signature() != self._snapshot_stat:
            raise StorageError('Поисковый снимок изменён.')

    def _stat_signature(self):
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def assert_snapshot(self):
        changed = self._stat_signature() != self._snapshot_stat if self._sealed else digest_file(self.path) != self.revision
        if changed:
            raise StorageError('Поисковый снимок изменён.')

    from contextlib import contextmanager

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path.as_uri() + '?mode=ro&immutable=1', uri=True)
        deadline = __import__('time').monotonic() + 5
        db.set_progress_handler(lambda: int(__import__('time').monotonic() >= deadline), 1000)
        try:
            yield db
        finally:
            db.close()

    def scope(self, request):
        if (request.corpus_id != self.manifest['corpus_id'] or
            request.corpus_revision not in (None, self.manifest['corpus_revision'])):
            raise UnknownId('Неизвестный корпус или ревизия.')
        self.assert_snapshot()
        return {k: self.manifest[k] for k in ('corpus_id','corpus_revision','index_revision','search_revision')}

    def search(self, request):
        result = self.scope(request)
        terms = list(dict.fromkeys(re.findall(r'[^\W_]+', request.query.casefold())))
        if not terms or len(terms) > 32:
            raise ValueError('Недопустимый запрос.')
        expression = ' OR '.join('"' + term + '"' for term in terms)
        with self.connection() as db:
            rows = db.execute('''SELECT d.doc_id,snippet(document_fts,1,'','',' … ',48),bm25(document_fts)
                FROM document_fts JOIN documents d ON d.rowid=document_fts.rowid
                WHERE document_fts MATCH ? ORDER BY bm25(document_fts),d.doc_id LIMIT ?''',
                (expression, request.limit)).fetchall()
        return {**result, 'hits': [dict(doc_id=row[0], snippet=row[1][:500], score=row[2], rank=i)
                                 for i, row in enumerate(rows, 1)]}

    def read(self, request):
        result = self.scope(request)
        if (request.doc_id is None) == (request.chunk_id is None):
            raise ValueError('Нужен ровно один идентификатор.')
        with self.connection() as db:
            doc_id, start, end = request.doc_id, 0, None
            if request.chunk_id is not None:
                row = db.execute('SELECT doc_id,start_char,end_char FROM chunks WHERE chunk_id=?', (request.chunk_id,)).fetchone()
                if row is None:
                    raise UnknownId('Неизвестный chunk_id.')
                doc_id, start, end = row
            row = db.execute('SELECT text FROM documents WHERE doc_id=?', (doc_id,)).fetchone()
        if row is None:
            raise UnknownId('Неизвестный doc_id.')
        text = row[0][start:end]
        if request.offset > len(text):
            raise ValueError('Offset за пределами текста.')
        fragment = text[request.offset:request.offset + request.limit]
        next_offset = request.offset + len(fragment)
        return {**result, 'doc_id': doc_id, 'chunk_id': request.chunk_id, 'text': fragment,
                'offset': request.offset, 'total_chars': len(text),
                'next_offset': next_offset if next_offset < len(text) else None}

    def chunks(self, request):
        result = self.scope(request)
        with self.connection() as db:
            if db.execute('SELECT 1 FROM documents WHERE doc_id=?', (request.doc_id,)).fetchone() is None:
                raise UnknownId('Неизвестный doc_id.')
            rows = db.execute('SELECT chunk_id FROM chunks WHERE doc_id=? ORDER BY start_char LIMIT ? OFFSET ?',
                              (request.doc_id, request.limit + 1, request.offset)).fetchall()
        return {**result, 'chunk_ids': [row[0] for row in rows[:request.limit]],
                'next_offset': request.offset + request.limit if len(rows) > request.limit else None}


def create_app(snapshot=None, sql_catalog=None):
    app = web.Application(client_max_size=64 * 1024)
    search = SearchStore(snapshot) if snapshot else None
    stores = {}
    pending = 0
    slots = asyncio.Semaphore(16)

    async def lifecycle(app):
        try:
            if sql_catalog:
                path = Path(sql_catalog).resolve()
                catalog = json.loads(path.read_text())
                if set(catalog) != {'version','databases'} or catalog['version'] != 1 or not catalog['databases']:
                    raise ValueError('Неверный каталог SQL.')
                for row in catalog['databases']:
                    if set(row) != {'db_id','path','revision'}:
                        raise ValueError('Неверная запись SQL-каталога.')
                    key = row['db_id'], row['revision']
                    if key in stores:
                        raise ValueError('Повторная БД/ревизия.')
                    stores[key] = SQLiteSnapshot(path.parent / row['path'], *key, timeout_seconds=5)
            yield
        finally:
            for store in stores.values():
                await store.aclose()
    app.cleanup_ctx.append(lifecycle)

    def resolve(identifier, revision):
        matches = [store for (i,r), store in stores.items() if (not identifier or i == identifier) and (not revision or r == revision)]
        if len(matches) != 1:
            if not matches:
                raise UnknownId('Неизвестная БД или ревизия.')
            raise ValueError('Укажите БД и ревизию.')
        return matches[0]

    async def handle(request):
        nonlocal pending
        if pending >= 128:
            return web.json_response({'error': {'code':'busy','message':'Очередь заполнена.'}}, status=503)
        pending += 1
        try:
            async with asyncio.timeout(5):
                async with slots:
                    route = request.path
                    if route == '/healthz':
                        result = {'status':'ok'}
                    elif route == '/v1/sql/databases':
                        result = {'databases':[dict(db_id=i,db_revision=r) for i,r in sorted(stores)]}
                    elif route == '/v1/sql/schema':
                        result = resolve(request.query.get('db_id'), request.query.get('db_revision')).info
                    elif route == '/v1/sql':
                        payload = SQLRequest.model_validate(await request.json())
                        store = resolve(payload.db_id, payload.db_revision)
                        result = {**(await store.query(payload.query)).model_dump(),
                                  'db_id': payload.db_id, 'db_revision': payload.db_revision}
                    elif search is None:
                        raise UnknownId('Поиск не настроен.')
                    elif route == '/v1/manifest':
                        result = search.manifest
                    else:
                        model, operation = {'/v1/search':(SearchRequest,search.search), '/v1/read':(ReadRequest,search.read),
                                            '/v1/chunks':(ChunksRequest,search.chunks)}[route]
                        payload = model.model_validate(await request.json())
                        result = await joined_thread(operation, payload)
                    return web.json_response(result)
        except TimeoutError:
            code, status = 'timeout', 504
        except UnknownId:
            code, status = 'unknown_id', 404
        except SqlQueryError:
            code, status = 'sql_query_error', 400
        except StorageError:
            code, status = 'storage_error', 500
        except (ValueError, TypeError, web.HTTPRequestEntityTooLarge):
            code, status = 'invalid_request', 400
        except (SearchError, sqlite3.Error, OSError):
            code, status = 'storage_error', 500
        finally:
            pending -= 1
        return web.json_response({'error': {'code':code,'message':'Операция не выполнена в установленных пределах.'}}, status=status)

    for route in ['/healthz','/v1/manifest','/v1/sql/schema','/v1/sql/databases']:
        app.router.add_get(route, handle)
    for route in ['/v1/search','/v1/read','/v1/chunks','/v1/sql']:
        app.router.add_post(route, handle)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--input', default='examples/fixtures/hydra/corpus.jsonl')
    build.add_argument('--output', default='indexes/hydra.sqlite')
    build.add_argument('--corpus-id', default='hydra-a2s4-kilt-pilot')
    build.add_argument('--revision')
    build.add_argument('--chunk-chars', type=int, default=2000)
    for command in ['serve','serve-sql']:
        serve = sub.add_parser(command)
        serve.add_argument('--snapshot', default=None if command == 'serve-sql' else 'indexes/hydra.sqlite')
        serve.add_argument('--sql-catalog')
        serve.add_argument('--host', default='127.0.0.1')
        serve.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    if args.command == 'build':
        print(json.dumps(build_snapshot(args.input,args.output,args.corpus_id,args.revision,args.chunk_chars)))
    else:
        if args.command == 'serve-sql' and not args.sql_catalog:
            parser.error('serve-sql требует --sql-catalog.')
        web.run_app(create_app(args.snapshot,args.sql_catalog), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
