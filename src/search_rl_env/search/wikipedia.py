"""Live English Wikipedia search with exact public response archives for replay."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import httpx

from search_rl_env.contracts import SearchHit

API_URL = 'https://en.wikipedia.org/w/api.php'
LIMITATION = ('Поиск по текущей английской Википедии через MediaWiki, не по срезу '
              'из 999 фрагментов. Это не фиксированный корпус KILT 2019 и не официальный '
              'HyDRA score. Полученные ревизии и полные ответы API сохраняются для replay.')


class WikipediaSearch:
    def __init__(self, client: httpx.AsyncClient, archive: Path):
        self.client = client
        self.archive = Path(archive)
        self.archive.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._cache = {}
        self.manifest = {
            'corpus_id': 'wikipedia-en-live', 'corpus_revision': 'recorded-run-' + uuid4().hex,
            'source': API_URL, 'index_algorithm': 'MediaWiki search',
            'revision_policy': 'live search; exact response bytes archived per query',
            'content_format': 'complete revision wikitext', 'corpus_limitation': LIMITATION,
            'archive': str(self.archive),
        }

    async def search(self, query, *, corpus_id, corpus_revision, limit):
        if (corpus_id != self.manifest['corpus_id'] or
                corpus_revision != self.manifest['corpus_revision']):
            raise ValueError('Unknown Wikipedia run snapshot.')
        if not query.strip() or not 1 <= limit <= 50:
            raise ValueError('Invalid Wikipedia search request.')
        params = {'action': 'query', 'format': 'json', 'formatversion': 2,
                  'generator': 'search', 'gsrsearch': query, 'gsrnamespace': 0,
                  'gsrlimit': limit, 'prop': 'revisions', 'rvprop': 'ids|timestamp|content',
                  'rvslots': 'main', 'rvlimit': 1, 'maxlag': 5}
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        # Serialize requests across episodes and cache identical queries within this run.
        async with self._lock:
            if key in self._cache:
                return list(self._cache[key])
            response = await self.client.get(API_URL, params=params, timeout=60,
                headers={'User-Agent': 'SearchRLEnv/0.1 (https://github.com/SergeyBarshin/search-rl-env)'})
            response.raise_for_status()
            raw = response.content
            digest = hashlib.sha256(raw).hexdigest()
            (self.archive / (key + '.response.json')).write_bytes(raw)
            (self.archive / (key + '.request.json')).write_text(json.dumps({
                'url': API_URL, 'params': params, 'response_sha256': digest,
                'retrieved_at': datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            data = response.json()
            if data.get('error') or data.get('warnings'):
                raise ValueError('Wikipedia returned an error or incomplete query warning; see archived response.')
            hits = []
            for page in sorted(data.get('query', {}).get('pages', []), key=lambda row: row['index']):
                revisions = page.get('revisions', [])
                if not revisions:
                    raise ValueError('Wikipedia result has no readable revision.')
                revision = revisions[0]
                text = revision['slots']['main'].get('content')
                if not isinstance(text, str):
                    raise ValueError('Wikipedia revision content is unavailable.')
                identifier = f"wikipedia:en:{page['pageid']}:{revision['revid']}"
                hits.append(SearchHit(doc_id=identifier, rank=len(hits) + 1, snippet=(
                    f"{page['title']}\nhttps://en.wikipedia.org/w/index.php?oldid={revision['revid']}\n"
                    f"Revision timestamp: {revision['timestamp']}\n\n{text}")))
            self._cache[key] = hits
            return list(hits)
