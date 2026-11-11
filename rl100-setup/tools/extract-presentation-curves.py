"""Read complete histories for the two completed 3D experiments without writes to runs."""
import json
from pathlib import Path
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

BASE=Path('/data1/suchenyu/RL-100/data')
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'rl100-setup/presentations/curves';OUT.mkdir(parents=True,exist_ok=True)
roots={'3D Diffusion':BASE/'outputs_vib_ablation/adroit_door_medium-rl100-0112_seed100/relu/dp3vib/skipnet',
       '3D Flow single':BASE/'outputs_flow/adroit_door_medium-rl100-flow-0112_seed100/relu/dp3vib/skipnet'}
output={}
for name,root in roots.items():
    runs=[]
    for p in sorted(root.glob('wandb/offline-run-*/run-*.wandb')):
        d=DataStore();d._fname=str(p);d._fp=p.open('rb');d._index=0;d._size_bytes=p.stat().st_size;d._opened_for_scan=True;d._read_header()
        rows=[];code=None
        while True:
            try:b=d.scan_data()
            except Exception:break
            if b is None:break
            r=wandb_internal_pb2.Record();r.ParseFromString(b)
            if r.HasField('exit'):code=r.exit.exit_code
            if r.HasField('history'):
                v={i.key or '.'.join(i.nested_key):json.loads(i.value_json) for i in r.history.item}
                rows.append({k:x for k,x in v.items() if isinstance(x,(int,float))})
        d._fp.close();runs.append({'file':str(p),'exit':code,'rows':rows})
        keys=sorted(set(k for v in rows for k in v if not k.startswith('_')))
        print(name,p.parent.name,'rows',len(rows),'exit',code,'KEYS',keys,flush=True)
    output[name]=runs
(OUT/'training-histories.json').write_text(json.dumps(output)+'\n')
