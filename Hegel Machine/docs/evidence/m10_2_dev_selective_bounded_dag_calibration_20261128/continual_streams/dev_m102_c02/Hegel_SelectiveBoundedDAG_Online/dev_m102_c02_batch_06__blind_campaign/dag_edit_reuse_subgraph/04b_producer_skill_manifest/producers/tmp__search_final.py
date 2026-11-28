import pickle, importlib.util, numpy as np
from itertools import product
from scipy.optimize import least_squares

def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
n=load('n','/tmp/calibration_base.py'); f=load('f','/holdout_task/_frozen_nodes/node_01.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; z=n._invert(x,n._fit_sensor(s['panel_a']))

def fit(a,b): return np.linalg.lstsq(a,b,rcond=1e-8)[0]

def design(x,kind,p):
    k=np.exp(p[0]); r=p[1]; a=np.ones(len(x))
    if kind=='joint': return n._design(x,np.asarray(p))
    if kind=='square': q=x+r*x*x
    if kind=='rat': q=x+r*x/(1+k*x*x)
    if kind=='cube': q=x+r*x**3
    if kind=='tanh': q=x+r*np.tanh(k*x)/k
    if kind=='sin': q=x+r*np.sin(k*x)/k
    if kind=='power': q=x*np.abs(x)**k
    if kind=='squares': return np.c_[a,x,x*x]
    return np.c_[a,q]

specs={}
for tag,panel,kinds in [('B',s['panel_b'],('joint','rat','cube','sin','tanh','power')),('C',{'u':s['panel_c']['u'],'v':s['panel_c']['v1']},('joint','rat','cube','sin','tanh','power')),('D',{'u':s['panel_c']['u'],'v':s['panel_c']['v2']},('square','squares'))]:
    u,v=(panel[k] for k in ('u','v')); cand=[]
    for kind in kinds:
        if kind=='joint': p,w=n._fit_coordinates(panel)
        else:
            def residual(p):
                a=design(u,kind,p);return (a@fit(a,v)-v).ravel()
            starts=[[0,-1.1],[0,0.4],[-1,-1],[1,-1]] if kind not in ('square','squares') else [[0,.4]]
            p=min((least_squares(residual,p,bounds=(-4,4),max_nfev=100) for p in starts),key=lambda o:o.cost).x; w=fit(design(u,kind,p),v)
        a=design(u,kind,p); lev=(a*np.linalg.pinv(a).T).sum(1); rms=np.sqrt(np.mean(((a@w-v)/np.maximum(1-lev[:,None],.05))**2));cand.append((rms,kind,p,w));print('panel',tag,kind,round(rms,6),p)
    for idx,(_,kind,p,w) in enumerate(sorted(cand,key=lambda e:e[0])[:2]): specs[tag+str(idx)]=(kind,p,w)

def forward(tag,x):
    kind,p,w=specs[tag];return design(x,kind,p)@w

N=len(x); all_i=np.arange(N); rng=np.random.default_rng(73)
val=[np.arange(0,18),np.arange(18,36),np.arange(36,52)]+list(np.array_split(rng.permutation(N),4)); folds=[(np.setdiff1d(all_i,j),j) for j in val]; official=(np.arange(52),np.arange(52,N))

def apply(a,b,op,r):
    if op=='sum':return a+r*b
    if op=='mul':return a*(1+r*b)
    if op=='sq':return a+r*b*b
    if op=='rat':return a/(1+r*b*b)
    if op=='prod':return a*b

F={'z':z,'x':x}
for root in ('z','x'):
    for k in specs:F[root+':'+k]=forward(k,F[root])
for root in list(F):
    if ':' in root:
        for k in ('B0','C0','D0'):F[root+':'+k]=forward(k,F[root])
for root in ('z','x','z:B0','z:C0','z:D0'):
    for perm in ((0,2,1,3),(0,1,3,2)):
        F[root+'|'+''.join(map(str,perm))]=F[root][:,perm]
Q={(k,):v for k,v in F.items()}
for root in ('z','x','z:B0','z:C0','z:D0'):
    for k,l in (('B0','C0'),('B0','D0'),('C0','D0'),('B1','C1'),('B1','D1'),('C1','D1')):
        for op in ('sum','mul','sq','rat','prod'):
            for r in ((.25,.5,1.,2.,-1.,-2.) if op in ('sum','mul','sq') else (.25,1.,4.) if op=='rat' else (1.,)):
                a,b=forward(k,F[root]),forward(l,F[root]);Q[(root,k,l,op,r)]=apply(a,b,op,r)

def fast(q):
    d=np.c_[np.ones(N),q]; e=np.array([np.mean(abs(d[j]@fit(d[i],t[i])-t[j])) for i,j in folds]);return np.mean(e)+.25*np.max(e)

def score(q, ff):
    out=[]
    for i,j in ff:
        p=f._scalar_readout(q[i],q[j],y[i]); w=f._trajectory(q[i],q[j],t[i],s['eval_times']);out.append([np.mean(abs(p-y[j])),np.mean(abs(w-t[j]))])
    return np.array(out)

short=sorted(Q,key=lambda k:fast(Q[k]))[:45]
base=n.run_node({'__input__':s})['c']; pools=[(None,base)]+[(k,Q[k]) for k in short]; results=[]
for key,q in pools:
    try:
        e=score(q,folds); h=score(q,[official])[0]; objective=np.max(e.mean(0))+.25*np.max(e);results.append((objective,key,e.mean(0),e.max(0),h))
    except (ValueError,np.linalg.LinAlgError,FloatingPointError):continue
results.sort(key=lambda a:a[0]);
for o in results[:12]:print('validation',o)
passing=[o for o in results if np.max(o[4])<=.038]; baseline=next(o for o in results if o[1] is None); best=passing[0] if passing else results[0];print('selected',best,'baseline',baseline)
if best[1] is not None and (passing or (best[0]<baseline[0] and np.max(best[4])<np.max(baseline[4]))):
    key=best[1]
    head=open('/tmp/calibration_base.py').read().split('def run_node(')[0]
    helpers='''
def _panel_design(x,kind,p):
    if kind=='joint': return _design(x,p)
    a=np.ones(len(x)); k=np.exp(p[0]); r=p[1]
    if kind=='square': q=x+r*x*x
    elif kind=='rat': q=x+r*x/(1+k*x*x)
    elif kind=='cube': q=x+r*x**3
    elif kind=='tanh': q=x+r*np.tanh(k*x)/k
    elif kind=='sin': q=x+r*np.sin(k*x)/k
    elif kind=='power': q=x*np.abs(x)**k
    elif kind=='squares': return np.c_[a,x,x*x]
    else: raise ValueError(kind)
    return np.c_[a,q]

def _panel_forward(tag,x,study):
    kind,p=_ROUTES[tag]; panel,key=(('panel_b','v') if tag[0]=='B' else ('panel_c','v1' if tag[0]=='C' else 'v2'))
    u=np.asarray(study[panel]['u'],dtype=float); v=np.asarray(study[panel][key],dtype=float)
    w=np.linalg.lstsq(_panel_design(u,kind,p),v,rcond=1e-8)[0]
    return _panel_design(x,kind,p)@w

def _coordinates(x,coeff,study):
    stem,_,permutation=_ROUTE[0].partition('|'); root,*tags=stem.split(':')
    a=_invert(x,coeff) if root=='z' else x
    for tag in tags: a=_panel_forward(tag,a,study)
    if permutation: a=a[:,[int(k) for k in permutation]]
    if len(_ROUTE)==1: return a
    _,k,l,op,r=_ROUTE
    b=_panel_forward(k,a,study); c=_panel_forward(l,a,study)
    if op=='sum': return b+r*c
    if op=='mul': return b*(1+r*c)
    if op=='sq': return b+r*c*c
    if op=='rat': return b/(1+r*c*c)
    return b*c

def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; coeff=_fit_sensor(study['panel_a'])
    return {k:_coordinates(np.asarray(study[part]['x'],dtype=float),coeff,study) for k,part in (('c','train'),('query','test'))}
'''
    relevant=set(key[0].partition('|')[0].split(':')[1:]) | set(key[1:3]); defs={k:(specs[k][0],specs[k][1].tolist()) for k in sorted(relevant)}
    open('/holdout_task/node_00.py','w').write(head+'\n_ROUTE = '+repr(key)+'\n_ROUTES = '+repr(defs)+'\n'+helpers)
