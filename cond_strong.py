"""Does a richer channel-context variable do better than the coarse 3-level one?"""
import sys, time, json, collections
sys.path.insert(0, r"C:\dev\swarm-prep"); sys.path.insert(0, r"C:\dev\swarm-sprint")
sys.path.insert(0, r"C:\dev\swarm-bench")
import numpy as np, swarm_te
from swarm_load import TopicSymboliser
import fastte, bench_gen
t0=time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

def prep(events, k=8, seed=0):
    ev=sorted(events,key=lambda e:e[1]); texts=[e[3] for e in ev]
    sym=TopicSymboliser(k=k,seed=seed).fit(texts).transform(texts)
    cnt=collections.Counter(e[0] for e in ev)
    agents=[a for a,n in cnt.most_common() if n>=20][:15]; keep=set(agents)
    stream=[(e[0],s) for e,s in zip(ev,sym) if e[0] in keep]
    n=len(stream); series={a:[swarm_te.SILENT]*n for a in agents}
    for t,(sp,s) in enumerate(stream): series[sp][t]=s
    coded,A=fastte.code_series(series)
    alpha=sorted({x for v in series.values() for x in v}); sil=alpha.index(swarm_te.SILENT)
    chan=[alpha.index(s) for _,s in stream]
    return coded,A,sil,chan

def modal_ctx(chan, sil, window=12):
    out=np.zeros(len(chan),dtype=np.int64)
    for t in range(len(chan)):
        lo=max(0,t-window); c=[x for x in chan[lo:max(lo,t-1)] if x!=sil]
        out[t]=collections.Counter(c).most_common(1)[0][0] if c else sil
    return out

res={}
for fam,spec in (("common_cause_null",dict(n_agents=12,n_messages=8000,edges=(),shared_topic=0.55)),
                 ("structure",dict(n_agents=12,n_messages=8000,shared_topic=0.55,
                                   edges=((0,5,0.4),(0,7,0.3),(2,9,0.5),(9,11,0.4))))):
    rec={}
    for s in range(3):
        inst=bench_gen.generate_instance(seed=900+s,**spec); gt=inst["ground_truth"]
        coded,A,sil,chan=prep(inst["events"],seed=s)
        ctx=modal_ctx(chan,sil)
        t=fastte.directed_tests_conditional(coded,ctx,A,A,sil,n_perm=100,seed=s)
        e={k for k,v in t.items() if v["p"]<0.05}
        rec.setdefault("edges",[]).append(len(e)); rec.setdefault("tp",[]).append(len(e&gt))
    res[fam]={"mean_edges":round(float(np.mean(rec["edges"])),2),
              "mean_tp":round(float(np.mean(rec["tp"])),2),"planted":len(gt)}
    log(f"{fam}: modal-context conditional -> mean edges {res[fam]['mean_edges']} tp {res[fam]['mean_tp']}/{len(gt)}")
json.dump(res,open("cond_strong.json","w"),indent=2)
log("done")
