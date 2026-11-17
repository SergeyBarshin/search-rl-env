import asyncio
import json
import sqlite3

from aiohttp.test_utils import TestServer
import pytest

from search_rl_env.search import SearchClient, SQLClientPool
from search_rl_env.search.local import digest_file
from search_rl_env.search.server import build_snapshot, create_app


@pytest.fixture
async def backend(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text(json.dumps({'doc_id':'doc-1','text':'перенос срока проекта'},ensure_ascii=False)+'\n')
    snapshot = tmp_path / 'search.sqlite'
    build_snapshot(corpus,snapshot,'integration',chunk_chars=5)
    databases=[]
    for name in ['left','right']:
        path = tmp_path / (name+'.sqlite')
        db=sqlite3.connect(path)
        db.execute('CREATE TABLE facts(value TEXT)')
        db.execute('INSERT INTO facts VALUES (?)',(name,))
        db.commit()
        db.close()
        databases.append({'db_id':name,'path':path.name,'revision':digest_file(path)})
    catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps({'version':1,'databases':databases}))
    async with TestServer(create_app(snapshot,catalog)) as server:
        yield str(server.make_url('/')).rstrip('/')


async def test_python_http_search_read_chunks(backend):
    async with SearchClient(backend) as client:
        hits=await client.search('срока',corpus_id='integration',corpus_revision=None,limit=10)
        assert [hit.doc_id for hit in hits] == ['doc-1']
        page=await client.chunks('doc-1')
        fragments=[await client.read(chunk_id=identifier) for identifier in page.chunk_ids]
        assert ''.join(item.text for item in fragments) == 'перенос срока проекта'


@pytest.mark.parametrize('concurrency',[16,32,64])
async def test_python_http_1000_sql_requests_route_to_own_database(backend,concurrency):
    async with SQLClientPool(backend,concurrency=concurrency) as pool:
        catalog=await pool.databases()
        bindings=[await pool.bind(row['db_id'],row['db_revision']) for row in catalog]
        source=iter(range(1000))
        completed=0
        async def worker():
            nonlocal completed
            for index in source:
                selected=index%len(bindings)
                result=await bindings[selected].query('SELECT value FROM facts')
                assert result.rows == [[catalog[selected]['db_id']]]
                completed+=1
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        assert completed == 1000
