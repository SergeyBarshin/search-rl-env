import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('hydra_pilot',ROOT/'examples/run_hydra_pilot10.py')
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_selection_uses_only_database_and_source_order_not_gold_answer():
    rows=[{'id':f'train_{i}','db_id':'hockey','final_question':f'Question {i}',
           'final_answer':f'Private {i}','sql_query':'SELECT 1'} for i in range(11)]
    rows.insert(0,{**rows[0],'id':'other','db_id':'other'})
    chosen=pilot.select_tasks(rows,{'db_id':'hockey','count':10})
    assert [r['id'] for r in chosen] == [f'train_{i}' for i in range(10)]
    for row in rows:row['final_answer']='Changed private answer'
    assert [r['id'] for r in pilot.select_tasks(rows,{'db_id':'hockey','count':10})] == [r['id'] for r in chosen]


def test_duplicate_questions_do_not_become_ten_distinct_tasks():
    rows=[{'id':str(i),'db_id':'hockey','final_question':'Same question',
           'final_answer':'answer','sql_query':'SELECT 1'} for i in range(10)]
    with pytest.raises(ValueError,match='10 разных'):
        pilot.select_tasks(rows,{'db_id':'hockey','count':10})


def test_html_report_escapes_data_and_shows_tools_environment_and_reference(tmp_path):
    from search_rl_env.reports.trajectories import write_episode_html,write_index
    result={'task_id':'real-question','task':{'question':'<script>bad()</script>'},
       'episode_id':'abc','model_metadata':{'model':'test'},'stop_reason':'finished',
       'usage':{'cost':None,'total_tokens':20},'answer':['Answer'],'selected_document_ids':['doc'],
       'sql_calls':1,'search_calls':1,'history':[{'role':'assistant','content':'choosing tools'}],
       'trajectory':[{'step':1,'model_duration_seconds':.1,'actions':[
        {'name':'sql','arguments':'{"query":"SELECT value FROM facts"}','duration_seconds':.01,
         'result':{'ok':True,'columns':['value'],'rows':[['<img src=x onerror=bad()>']]}},
        {'name':'search','arguments':'{"query":"value"}','duration_seconds':.01,
         'result':{'ok':True,'hits':[{'doc_id':'doc','rank':1,'snippet':'Real text'}]}}]}]}
    target=tmp_path/'abc.html'
    write_episode_html(result,{'db_id':'hockey','sql_schema':'CREATE TABLE facts(value TEXT);',
                               'corpus_limitation':'Limited corpus'}, {'final_answer':'PRIVATE_REFERENCE'},target)
    text=target.read_text()
    assert '<script>bad()' not in text and '&lt;script&gt;' in text
    assert '<img src=x' not in text
    assert 'SELECT value FROM facts' in text and 'Real text' in text
    assert 'PRIVATE_REFERENCE' in text and 'Эталон для ручного разбора' in text
    assert 'нет данных' in text and 'CREATE TABLE facts' in text
    write_index([{'task_id':'real-question','question':'<script>bad()</script>',
                  'status':'finished','html':'abc.html'}],tmp_path/'index.html',{'protocol':'test'})
    assert '<script>bad()' not in (tmp_path/'index.html').read_text()


def test_rebuild_report_uses_saved_episode_and_verdict_without_model(tmp_path):
    import json

    review = tmp_path / 'html'
    review.mkdir()
    episode = {'schema_version': '3', 'task_id': 'a', 'task': {'question': 'Saved question'},
               'episode_id': 'first', 'stop_reason': 'finished', 'answer': ['Saved answer']}
    manifest = {'episodes': [{'episode_id': 'first', 'task_id': 'a', 'status': 'finished',
                              'question': 'Saved question', 'html': 'first.html'}],
                'sources': {'task_ids': ['a', 'b']}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (review / 'first.json').write_text(json.dumps(episode))
    (review / 'first.environment.json').write_text('{}')
    verdicts = tmp_path / 'evaluation' / 'verdicts'
    verdicts.mkdir(parents=True)
    (verdicts.parent / 'judge.json').write_text('{}')
    (verdicts / 'first.json').write_text(json.dumps({
        'episode_id': 'first', 'task_id': 'a', 'score': 1, 'judge_called': True,
        'usage': {'total_tokens': 12}, 'duration_seconds': 0.1,
    }))

    assert pilot.rebuild_report(tmp_path) == 0
    assert 'Saved answer' in (review / 'first.html').read_text()
    assert 'Saved question' in (review / 'index.html').read_text()
    summary = json.loads((verdicts.parent / 'summary.json').read_text())
    assert summary['accuracy_bounds'] == [0.5, 1.0]
    assert summary['judge_usage']['total_tokens'] == 12
