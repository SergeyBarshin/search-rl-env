import json
from types import SimpleNamespace
import sqlite3

import pytest

from search_rl_env.search.local import digest_file
from search_rl_env.search.server import build_snapshot, create_app, SearchStore, SearchRequest, ReadRequest


def test_python_bm25_unicode_read_chunks_and_conflicts(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text(json.dumps({'doc_id':'one','text':'перенос срока проекта'},ensure_ascii=False)+'\n')
    path = tmp_path / 'search.sqlite'
    first = build_snapshot(corpus,path,'demo',chunk_chars=5)
    assert build_snapshot(corpus,path,'demo',chunk_chars=5) == first
    store = SearchStore(path)
    hits = store.search(SearchRequest(corpus_id='demo',query='срока'))['hits']
    assert hits[0]['doc_id'] == 'one'
    result = store.read(ReadRequest(corpus_id='demo',doc_id='one'))
    assert result['text'] == 'перенос срока проекта'
    with pytest.raises(ValueError):
        build_snapshot(corpus,path,'other')


async def test_python_http_handlers_sql_catalog_and_errors(tmp_path):
    path = tmp_path / 'facts.sqlite'
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE facts(value TEXT); INSERT INTO facts VALUES ('ok');")
    db.close()
    revision = digest_file(path)
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'version':1,'databases':[{'db_id':'facts','path':path.name,'revision':revision}]}))
    app = create_app(sql_catalog=catalog)
    app.freeze()
    await app.startup()
    try:
        handlers = {route.resource.canonical:route.handler for route in app.router.routes()}
        async def request(route, payload=None, query=None):
            async def read(): return payload
            result = await handlers[route](SimpleNamespace(path=route,query=query or {},json=read))
            return result.status,json.loads(result.text)
        status, result = await request('/v1/sql',{'db_id':'facts','db_revision':revision,'query':'SELECT value FROM facts'})
        assert status == 200 and result['rows'] == [['ok']]
        status,_ = await request('/v1/sql',{'db_id':'facts','db_revision':revision,'query':"SELECT file FROM 'pragma_database_list'"})
        assert status == 400
        status,_ = await request('/v1/sql/schema', query={'db_id':'missing'})
        assert status == 404
    finally:
        await app.cleanup()
