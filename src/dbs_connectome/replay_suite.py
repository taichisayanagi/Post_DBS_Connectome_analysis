"""Allowlisted numerical replay suite; never an image-to-paper certificate."""
from datetime import datetime,timezone
import html
from pathlib import Path
import re
import time

from .controls import run_controls
from .comparative import run_comparative
from .lateralized import run_lateralized
from .exploratory import run_exploratory
from .graphs import run_graph_study
from .geometry_controls import run_geometry_controls
from .representations import run_representations
from .reference_validation import run_reference_validation
from .sensitivity import run_sensitivity
from .study import manifest_paths,run_study
from .provenance import (assert_inputs_unchanged,digest,freeze_inputs,read_json,
                         record,require_separate_output,write_json)

OPERATIONS={
    'study':(run_study,'study_result.json'),
    'graph-study':(run_graph_study,'graph_result.json'),
    'sensitivity-study':(run_sensitivity,'sensitivity_result.json'),
    'control-study':(run_controls,'control_result.json'),
    'reference-validation':(run_reference_validation,'reference_validation_result.json'),
    'geometry-controls':(run_geometry_controls,'geometry_result.json'),
    'representation-study':(run_representations,'representation_result.json'),
    'comparative-study':(run_comparative,'comparative_result.json'),
    'lateralized-study':(run_lateralized,'lateralized_result.json'),
    'exploratory-study':(run_exploratory,'exploratory_result.json'),
}


def suite_paths(spec):
    if spec.get('schema_version')!=1 or spec.get('scope')!='numerical_replay':
        raise ValueError('Suite scope must be numerical_replay, not clinical validation')
    tasks=spec.get('tasks',[])
    if not tasks or len(tasks)>32:
        raise ValueError('Specify 1..32 allowlisted numerical replay tasks')
    ids=[];paths=[]
    for task in tasks:
        name=task.get('id','')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}',name) or name in ids:
            raise ValueError('Unique safe task IDs required')
        ids.append(name)
        if task.get('operation') not in OPERATIONS:
            raise ValueError('Unsupported suite operation; arbitrary commands are forbidden')
        if digest(task['manifest'])!=task.get('manifest_sha256'):
            raise ValueError('Task manifest changed after suite review')
        paths.extend([task['manifest'],*manifest_paths(read_json(task['manifest']))])
    return paths


def run_suite(config,out):
    out=Path(out);spec=read_json(config);paths=[config,*suite_paths(spec)]
    for path in paths:require_separate_output(out,Path(path).parent)
    frozen=freeze_inputs(paths);tasks=spec['tasks'];rows=[]
    for i,task in enumerate(tasks,1):
        print(f'[step suite {i}/{len(tasks)}] {task["id"]}: {task["operation"]}',flush=True)
        target=out/task['id'];target.mkdir(mode=0o700)
        operation,resultfile=OPERATIONS[task['operation']]
        start=time.monotonic();error=None;result={}
        try:
            operation(task['manifest'],target)
            result=read_json(target/resultfile)
            checks=result.get('checks',result.get('expected_checks',[]))
            status='verified_numeric_replay' if checks and all(v['passed'] for v in checks) else 'computed_not_frozen_verified'
        except Exception as exc:
            status='failed';error=f'{type(exc).__name__}: {exc}'
            if (target/resultfile).is_file():result=read_json(target/resultfile)
        rows.append({'id':task['id'],'operation':task['operation'],'status':status,
                     'elapsed_seconds':round(time.monotonic()-start,3),'error':error,
                     'result':str(target/resultfile) if (target/resultfile).exists() else None,
                     'checks':result.get('checks',result.get('expected_checks',[]))})
        write_json(out/f'task_{i:02d}_status.json',rows[-1])
        # A failure does not suppress independent checks in subsequent tasks.
    assert_inputs_unchanged(frozen)
    failed=any(r['status']=='failed' for r in rows)
    verified=all(r['status']=='verified_numeric_replay' for r in rows)
    result={'scope':'numerical_replay','status':'failed' if failed else ('verified_numeric_replay' if verified else 'completed_with_unverified_tasks'),
            'utc':datetime.now(timezone.utc).isoformat(),'tasks':rows,'source_bytes_unchanged':True,
            'full_image_to_paper_validated':False,
            'warning':'This suite checks only the explicitly supplied numerical analyses. Missing image reconstruction, mask/atlas expert QC, tractography, supplementary analyses and figure validation are not implied.'}
    write_json(out/'suite_result.json',result)
    content=''.join('<tr><td>'+html.escape(r['id'])+'</td><td>'+html.escape(r['operation'])+'</td><td>'+html.escape(r['status'])+'</td><td>'+str(r['elapsed_seconds'])+' s</td><td>'+html.escape(r['error'] or '')+'</td></tr>' for r in rows)
    (out/'suite_review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Numerical replay</title><style>body{font:15px system-ui;background:white;color:#222;margin:28px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse}p{max-width:900px}</style><h1>Numerical replay</h1><p>'+html.escape(result['warning'])+'</p><p>Full image-to-paper validation: <strong>not certified</strong></p><table><thead><tr><th>Task</th><th>Operation</th><th>Result</th><th>Time</th><th>Error</th></tr></thead><tbody>'+content+'</tbody></table></html>',encoding='utf-8')
    record(out,'replay-suite',paths,{'scope':'numerical_replay','task_count':len(tasks)},[out/'suite_result.json',out/'suite_review.html'],result['status'])
    if failed:raise ValueError('One or more numerical replay tasks failed; inspect suite_result.json')
