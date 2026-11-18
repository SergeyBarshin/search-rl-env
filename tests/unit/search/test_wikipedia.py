import hashlib
import json

import httpx
import pytest

from search_rl_env.search.wikipedia import WikipediaSearch


async def test_full_revision_results_are_ranked_archived_and_cached(tmp_path):
    calls=[]
    text='A full document ' * 1000
    def handler(request):
        calls.append(request)
        assert request.url.params['gsrsearch']=='Jacques Plante'
        assert 'rvlimit' not in request.url.params
        return httpx.Response(200,json={'query':{'pages':[
            {'index':2,'pageid':20,'title':'Second','revisions':[{'revid':22,
                'timestamp':'2026-01-01T00:00:00Z','slots':{'main':{'content':'Other'}}}]},
            {'index':1,'pageid':10,'title':'First','revisions':[{'revid':11,
                'timestamp':'2026-01-01T00:00:00Z','slots':{'main':{'content':text}}}]},
        ]}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend=WikipediaSearch(client,tmp_path)
        scope={key:backend.manifest[key] for key in ('corpus_id','corpus_revision')}
        hits=await backend.search('Jacques Plante',limit=10,**scope)
        assert hits[0].doc_id=='wikipedia:en:10:11'
        assert hits[0].snippet.endswith(text)
        assert [hit.rank for hit in hits]==[1,2]
        assert await backend.search('Jacques Plante',limit=10,**scope)==hits
        assert len(calls)==1
    request=json.loads(next(tmp_path.glob('*.request.json')).read_text())
    raw=next(tmp_path.glob('*.response.json')).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==request['response_sha256']


async def test_remote_warning_is_archived_and_never_replaced_with_fixture(tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:
            httpx.Response(200,json={'warnings':{'revisions':'incomplete'}}))) as client:
        backend=WikipediaSearch(client,tmp_path)
        with pytest.raises(ValueError,match='warning'):
            await backend.search('A question',corpus_id=backend.manifest['corpus_id'],
                corpus_revision=backend.manifest['corpus_revision'],limit=10)
        assert len(list(tmp_path.glob('*.response.json')))==1
        with pytest.raises(ValueError,match='snapshot'):
            await backend.search('A question',corpus_id='other',corpus_revision='old',limit=10)
