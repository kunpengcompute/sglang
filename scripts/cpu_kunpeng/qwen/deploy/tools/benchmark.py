"""One short shape warmup plus one exact-token formal round; keep all raw events."""
import argparse,json,subprocess,sys,time,uuid
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
for name in ['input-tokens','output-tokens','concurrency','warmup-tokens']:
    p.add_argument('--'+name,type=int,required=True)
a=p.parse_args()
assert min(a.input_tokens,a.output_tokens,a.concurrency,a.warmup_tokens)>0
state=json.loads((a.run/'STATE.json').read_text());assert state['phase']=='READY'
port=state['url'].rsplit(':',1)[1]
out=a.run/'benchmarks'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
logs=Path(state['logs']); log=logs/'rank0.log'
def captures():
    return [x for x in log.read_text(errors='replace').splitlines() if '[graph] captured ' in x]
record={'mtp_steps':state.get('mtp_steps',0),'mtp':state['mtp'],'phase':'starting','rounds':[],'note':'Warmup follows startup smoke request; not a process-cold benchmark.'}
def save():
    (out/'SUMMARY.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    for label,n in [('shape_warmup',a.warmup_tokens),('measured_once',a.output_tokens)]:
        before=captures();report=out/(label+'.json');record['phase']=label;save()
        log_dir=logs/'benchmarks'/out.name; log_dir.mkdir(parents=True,exist_ok=True)
        with (log_dir/(label+'.log')).open('w') as f:
            subprocess.run([sys.executable,'-u','-B','-S',str(Path(__file__).with_name('bench_requested.py')),
                '--input-tokens',str(a.input_tokens),'--output-tokens',str(n),'--concurrency',str(a.concurrency),
                '--port',port,'--output',str(report)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2100)
        d=json.loads(report.read_text());requests=d['requests']
        assert d['status']=='complete' and len(requests)==a.concurrency
        assert all(x['sse_done'] and not x['error'] and len(x['output_ids'])==n and x['prompt_tokens_reported']==a.input_tokens and x['cached_tokens_reported']==0 for x in requests)
        acceptance = None
        if state['mtp']:
            verify_count = sum(x['meta_info']['spec_verify_ct'] for x in requests)
            accepted = sum(x['meta_info']['spec_accepted_drafts'] for x in requests)
            assert verify_count > 0, 'MTP request did not report verification'
            steps = state['mtp_steps']
            acceptance = {'verify_count': verify_count, 'accepted_draft_tokens': accepted,
                'proposed_draft_tokens': verify_count * steps,
                'draft_acceptance_rate': accepted / (verify_count * steps),
                'output_tokens_per_verify': sum(len(x['output_ids']) for x in requests) / verify_count,
                'definition': 'accepted/(verify_count*steps); output_tokens_per_verify includes target tokens'}
        new=captures()[len(before):]
        record['rounds'].append({'label':label,'input_tokens':a.input_tokens,'output_tokens':n,'concurrency':a.concurrency,
             'ttft_ms':d['ttft_ms'],'tpot_ms':d['tpot_ms'],'throughput_output_tokens_per_s':d['throughput_output_tokens_per_s'],
             'new_captures_rank0':new,'execution_class':'capture_and_replay' if new else 'no_new_capture_observed',
             'frequency_condition_at_end':json.loads((a.run/'STATE.json').read_text()).get('frequency'),
             'mtp_acceptance':acceptance,'report':report.name,'requests_verified':len(requests)})
        save()
    record['phase']='complete'
except BaseException as e:
    record.update(phase='failed',error=repr(e));raise
finally:save()
print(json.dumps({'output':str(out),'summary':record},ensure_ascii=False,indent=2))
