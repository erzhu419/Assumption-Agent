exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from pathlib import Path
import contextlib, io
root=Path('/holdout_task')
common='''
def components(z, b, c, p, spec):
    a, family, warp, length, order, mix = spec
    if family == 0: X=np.c_[a*z+b,a*z-b]
    elif family == 1: X=np.c_[a*z,b]
    elif family == 2: X=a*z+b
    elif family == 3: X=a*z-b
    elif family == 4: X=np.c_[a*z+b,a*z-b,.24*c,.12*p]
    else: X=np.c_[a*z+b,a*z-b,a*z]
    X=X/max(np.sqrt(np.mean(X[:1]*0)+np.var(X)),1e-8)
    if warp: X=np.arcsinh(warp*X)/warp
    dot=X@X.T/X.shape[1]
    df=(X[:,None,:]-X[None,:,:])**2
    G=np.exp(-length*df)
    d=X.shape[1]
    M=G.mean(2)
    e2=(d*M*M-(G*G).mean(2))/(d-1)
    e3=((d*M)**3-3*d*d*M*(G*G).mean(2)+2*d*(G**3).mean(2))/(d*(d-1)*(d-2))
    e4=((d*M)**4-6*(d*M)**2*d*(G*G).mean(2)+3*(d*(G*G).mean(2))**2+8*d*d*M*(G**3).mean(2)-6*d*(G**4).mean(2))/(d*(d-1)*(d-2)*(d-3))
    if order == 0: return 1+dot+np.exp(-length*df.mean(2))
    if order == 1: return 1+dot+M+mix*e2
    if order == 2: return 1+dot+M+mix*e2+mix**2*e3
    if order == 3: return 1+dot+M+mix*e2+mix**2*e3+mix**3*e4
    if order == 4: return (1+dot)**2+M+mix*e2
    if order == 5: return (1+dot)**3+np.exp(-length*df.mean(2))
    if order == 6: return (1+dot)**2+np.exp(-length*df.mean(2))
    vals=X[:,:,None]*X[:,None,:]; vals=vals[:,np.triu_indices(d)[0],np.triu_indices(d)[1]]
    pw=vals@vals.T/vals.shape[1]
    if order==7: return 1+dot+pw+mix*((1+dot)**3+M+e2)
    gg=np.exp(-length*(vals[:,None,:]-vals[None,:,:])**2).mean(2)
    return 1+dot+M+mix*e2+gg

def predict_kernel(z,b,c,p,target,models):
    from scipy.linalg import cho_factor, cho_solve
    n=len(target); ntest=len(z)-n
    if not ntest: return np.empty((0,)+target.shape[1:])
    out=0.
    for spec,ridge,weight in models:
        K=components(z,b,c,p,spec)
        out=out+weight*(K[n:,:n]@cho_solve(cho_factor(K[:n,:n]+ridge*np.eye(n)),target))
    return out
'''
# Train-only scaling, so predictions are independent of test cohort composition.
common=common.replace('p, spec):','p, spec, n):').replace('np.mean(X[:1]*0)+np.var(X)','np.var(X[:n])').replace('p,spec)\n','p,spec,n)\n')
exec(common)
sys.path.insert(0,str(root)); import node_00 as m
z,b,c,p=m.features(s); Y=T
from scipy.linalg import cho_factor,cho_solve
from itertools import product
sl=sl+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(129).permutation(n),4)]
Xs=[tuple(a[np.r_[i,j]] for a in (z,b,c,p)) for i,j in sl]
I=[np.eye(len(i)) for i,j in sl]
records=[]
def ev(spec,rdgs):
    ks=[components(*a,spec,len(i)) for a,(i,j) in zip(Xs,sl)]
    for r in rdgs:
        P=[]; e=[]
        for k,(i,j),eye in zip(ks,sl,I):
            l=len(i); q=k[l:,:l]@cho_solve(cho_factor(k[:l,:l]+r*eye,check_finite=False),Y[i],check_finite=False); P.append(q); diff=q-Y[j]; e.append([abs(diff[:,0]).mean(),abs(diff[:,1:]@f.T).mean()])
        records.append((spec,r,np.array(e),P))
for a,warp,l,o in product([.08,.17,.27,.37,.52,.75,1.15],[0.,.65],[.012,.045,.16,.55,1.8],range(9)):
    ev((a,0,warp,l,o,.75),[1e-7,1e-5,.0004,.005])
for fam,a,l,o in product(range(1,6),[.22,.45,.85],[.035,.18,.8],[1,2,4,5,6,8]):
    ev((a,fam,0.,l,o,.75),[1e-7,.0001,.004])
for h in range(2):
    rank=sorted(records,key=lambda r: .35*r[2][0,h]+.45*r[2][1:,h].mean()+.2*r[2][1:,h].max()); print('top',h,[(r[0],r[1],r[2][:,h].round(5)) for r in rank[:4]],flush=True)
    candidates=[]
    def score(pred):
        es=[q-Y[j] for q,(_,j) in zip(pred,sl)]
        E=np.array([abs(e[:,0]).mean() if h==0 else abs(e[:,1:]@f.T).mean() for e in es]); return .35*E[0]+.45*E[1:].mean()+.2*E[1:].max(),E
    for k in [1,3,6,12,25,60]:
        chosen=[]; seen=set()
        for rec in rank:
            if rec[0] in seen: continue
            chosen.append(rec);seen.add(rec[0]);
            if len(chosen)==k:break
        P=[np.mean([r[3][i] for r in chosen],axis=0) for i in range(len(sl))]
        val,e=score(P); candidates.append((val,e,[(r[0],r[1],1/k) for r in chosen]))
    for start in range(4):
        chosen=[rank[start]]; P=rank[start][3]; best=score(P)[0]; candidates.append((best,score(P)[1],[(chosen[0][0],chosen[0][1],1.)]))
        for step in range(7):
            w=1/(step+2); trial=min(rank[:100],key=lambda r:score([(1-w)*x+w*y for x,y in zip(P,r[3])])[0]); PP=[(1-w)*x+w*y for x,y in zip(P,trial[3])]; val,e=score(PP)
            if val>=best: break
            P=PP; best=val;chosen.append(trial); candidates.append((val,e,[(r[0],r[1],w) for r in chosen]))
    v,e,models=min(candidates,key=lambda r:r[0]); print('chosen',h,e.round(6),models,flush=True)
    Path('/tmp/finalmodels'+str(h)).write_text(repr(models))
base=(root/'node_00.py').read_text().split('\ndef kernel')[0]
mods=[Path('/tmp/finalmodels'+str(h)).read_text() for h in range(2)]
(root/'node_00.py').write_text(base+common+'\nMODELS='+mods[0]+'''

def run_node(inputs: dict) -> dict:
    study=inputs['__input__']
    z,b,c,p=features(study)
    yhat=predict_kernel(z,b,c,p,np.asarray(study['train']['y']),MODELS)
    return {'y_hat':yhat,'z':z,'b':b,'c':c,'p':p}
''')
(root/'node_02.py').write_text('import numpy as np\n'+common+'\nMODELS='+mods[1]+'''

def run_node(inputs: dict) -> dict:
    a=inputs['node_00']; t=inputs['node_01']
    weights=predict_kernel(a['z'],a['b'],a['c'],a['p'],t['modes'],MODELS)
    return {'state_pred':weights@t['basis'].T}
''')
