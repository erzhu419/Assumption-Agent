import runpy, sys
from time import perf_counter
ns=runpy.run_path('/tmp/test_robin_eq01.py'); m=ns['m']; gen=ns['generate']
from threadpoolctl import threadpool_info
print(threadpool_info(), flush=True)
for kind in map(int,sys.argv[1:]):
    calls={k:[] for k in ('_factor','cg','_profile')}
    def wrap(name, fn):
        def tagged(*a,**kw):
            start=perf_counter(); out=fn(*a,**kw); sec=perf_counter()-start
            calls[name].append(sec)
            if name=='_factor': print(f'factor {sec:.3f}s',flush=True)
            return out
        return tagged
    saved={name:getattr(m,name) for name in calls}
    for name,fn in saved.items(): setattr(m,name,wrap(name,fn))
    s=gen(975+kind,(28,28,28),kind)
    start=perf_counter(); m.solve_robin_continuation_tomography(s)
    print(kind,perf_counter()-start,{k:(len(v),sum(v),max(v,default=0)) for k,v in calls.items()},flush=True)
    for name,fn in saved.items(): setattr(m,name,fn)
