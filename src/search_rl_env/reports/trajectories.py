"""Читаемый HTML на эпизод, без сервера, внешних ресурсов и credentials."""
from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote


def esc(value):
    return html.escape(str(value), quote=True)


def pretty(value):
    return '<pre>' + esc(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)) + '</pre>'


def details(label, body):
    return '<details><summary>' + esc(label) + '</summary>' + body + '</details>'


def table(columns, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+esc(c)+'</th>' for c in columns) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc('NULL' if v is None else v)+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def number(value, suffix=''):
    return 'нет данных' if value is None else f'{value:.4g}{suffix}'


STYLE = '''
:root{color-scheme:light dark;--bg:#f4f6fa;--card:#fff;--text:#182133;--muted:#536178;--line:#d6ddea;--accent:#2154b8}
@media(prefers-color-scheme:dark){:root{--bg:#141923;--card:#1c2432;--text:#e4eaf4;--muted:#acb9cf;--line:#38445a;--accent:#9cbaff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:30px 24px}a{color:var(--accent)}h1{font-size:28px;line-height:1.3}h2{font-size:21px}h3{font-size:17px;margin:0 0 12px}
.card,article{padding:22px;background:var(--card);border:1px solid var(--line);border-radius:12px;margin:18px 0}.muted{color:var(--muted)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.fact{padding:12px;border:1px solid var(--line);border-radius:8px}.fact b{display:block}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:600px;overflow:auto;padding:14px;border:1px solid var(--line);border-radius:8px;font:13px/1.55 ui-monospace,monospace}
code{overflow-wrap:anywhere}summary{cursor:pointer;color:var(--accent);padding:10px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:8px 12px;border:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}.scroll{overflow:auto;max-height:500px}
.step{border-left:4px solid var(--accent);padding-left:18px;margin:24px 0}.tag{display:inline-block;padding:3px 9px;border:1px solid var(--line);border-radius:20px;font-size:13px;margin-right:8px}.hit{padding:12px 0;border-bottom:1px solid var(--line)}.question{font-size:20px}.warning{border-left:4px solid #ba8435;padding:12px 18px}.answer{font-size:18px;white-space:pre-wrap}
@media(max-width:600px){main{padding:16px 12px}.card,article{padding:15px}h1{font-size:23px}}
'''


def page(title, body):
    return '<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+STYLE+'</style></head><body><main>'+body+'</main></body></html>'


def action_view(action):
    name, observation = action['name'], action['result']
    body = '<h3>'+esc(name)+' <span class="muted">'+number(action.get('duration_seconds'),' с')+'</span></h3>'
    try:
        arguments=json.loads(action['arguments'])
    except (ValueError, TypeError):
        arguments=action['arguments']
    body += '<b>Аргументы, выбранные агентом</b>'+pretty(arguments)
    if name == 'sql' and observation.get('ok') and 'rows' in observation:
        body += '<b>Наблюдение из настоящей SQLite</b>'+table(observation['columns'],observation['rows'])
        if observation.get('truncated'):
            body += '<p class="warning">Результат обрезан лимитом SQL-инструмента.</p>'
    elif name == 'search' and observation.get('ok'):
        body += '<b>Документы, показанные агенту</b>'
        for hit in observation.get('hits',[]):
            body += '<div class="hit"><b>'+esc(hit['rank'])+'. '+esc(hit['doc_id'])+'</b><p>'+esc(hit['snippet'])+'</p></div>'
        if not observation.get('hits'):
            body += '<p>Пустая выдача.</p>'
    else:
        body += '<b>Ответ инструмента</b>'+pretty(observation)
    return '<article>'+body+'</article>'


def write_episode_html(result, environment, reference, destination):
    r = result.model_dump(mode='json') if hasattr(result,'model_dump') else result
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    model=r.get('model_metadata',{}).get('model','неизвестно')
    body='<a href="index.html">← Все вопросы</a><h1>'+esc(r['task_id'])+'</h1>'
    body+='<div class="card question">'+esc(r['task']['question'])+'</div>'
    facts={'Модель':model,'Остановка':r['stop_reason'],'Время эпизода':number(r.get('duration_seconds'),' с'),
           'SQL / search':f"{r.get('sql_calls',0)} / {r.get('search_calls',0)}",'Токены':r.get('usage',{}).get('total_tokens'),
           'Cost из usage API':number(r.get('usage',{}).get('cost'),' USD')}
    body+='<div class="facts">'+''.join('<div class="fact"><b>'+esc(k)+'</b>'+esc('нет данных' if v is None else v)+'</div>' for k,v in facts.items())+'</div>'
    source = {
        'wikipedia': 'поиск по английской Википедии с архивом ревизий',
        'kilt_hybrid': 'локальный поиск BM25 + FAISS по снимку KILT',
    }.get(environment.get('search_backend'), 'поиск BM25 по текстам KILT')
    body+='<section class="card"><h2>Среда, доступная агенту</h2><p>Настоящая SQLite '+esc(environment.get('db_id',''))+' и '+esc(source)+'. Постоянная память агента выключена.</p>'
    body+='<p class="warning">'+esc(environment.get('corpus_limitation',''))+'</p>'
    body+=details('Публичная схема SQLite', '<pre>'+esc(environment.get('sql_schema',''))+'</pre>')
    body+=details('Снимки, ревизии, размеры и ограничения',pretty(environment))+'</section>'
    body+='<h2>Траектория: что делал агент</h2><p class="muted">Вызовы приведены из сохранённой траектории. Текст и рассуждения модели не восстанавливаются и не дописываются.</p>'
    assistants=[m for m in r.get('history',[]) if m.get('role')=='assistant']
    for index,step in enumerate(r.get('trajectory',[])):
        body+='<section class="step"><h2>Ход '+esc(step['step'])+' · модель '+number(step.get('model_duration_seconds'),' с')+'</h2>'
        if index < len(assistants):
            message=assistants[index]
            if message.get('content'):
                body+='<article><h3>Текст ответа модели</h3><p>'+esc(message['content'])+'</p></article>'
            body+=details('Сохранённый ответ модели и usage',pretty({'message':message,'usage':step.get('usage')}))
        for action in step.get('actions',[]):
            body+=action_view(action)
        if step.get('error'):
            body+='<div class="warning">'+esc(step['error']['code'])+': '+esc(step['error']['message'])+'</div>'
        body+='</section>'
    body+='<section class="card"><h2>Итог агента</h2><div class="answer">'+esc('\n'.join(r.get('answer') or ['Финальный ответ не получен.']))+'</div>'
    body+='<p>Выбранные источники: '+esc(', '.join(r.get('selected_document_ids',[])))+'</p>'
    if r.get('error'):body+=pretty(r['error'])
    body+='</section>'
    if reference is not None:
        body+=details('Эталон для ручного разбора', pretty(reference))
    body+=details('Полный сохранённый контекст модели',pretty(r.get('history',[])))
    body+=details('Промпт и бюджеты',pretty(r.get('config',{})))
    body+='<p><a href="'+quote(destination.with_suffix('.json').name)+'">Исходный EpisodeResult JSON</a></p>'
    destination.write_text(page(r['task_id']+' · траектория',body),encoding='utf-8')
    return destination


def write_index(entries,destination,run_info):
    body='<h1>Реальные hybrid-вопросы · траектории агента</h1><p>SQL + поиск документов, фиксированные веса, без постоянной памяти.</p>'
    finished=sum(entry['status']=='finished' for entry in entries)
    body+='<section class="card"><h2>Результат прогона</h2><p>Сохранено эпизодов: '+str(len(entries))+'. Завершены через finish: '+str(finished)+'.</p>'
    if run_info.get('billing')=='internal':
        body+='<p class="muted">Использован режим internal, разрешённый как API без денежного биллинга. Значения cost сообщает API; они не подтверждают фактическое списание денег.</p>'
    if entries and finished != len(entries):
        body+='<p class="warning">У '+str(len(entries)-finished)+' эпизодов нет финального ответа. Завершение работы runner не означает успешное решение задач.</p>'
    body+='</section>'
    body+=details('Протокол прогона и ограничения источников',pretty(run_info))
    for entry in entries:
        body+='<article><span class="tag">'+esc(entry['status'])+'</span><h2><a href="'+quote(entry['html'])+'">'+esc(entry['task_id'])+'</a></h2><p>'+esc(entry['question'])+'</p></article>'
    Path(destination).write_text(page('Hybrid pilot · вопросы и траектории',body),encoding='utf-8')
