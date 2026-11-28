exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from pathlib import Path
from scipy.linalg import cho_factor,cho_solve
from itertools import product
sys.path.insert(0,'/holdout_task');import node_00 as a,node_02 as b
z,v,c,p=a.features(s); folds=sl+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(129).permutation(n),4)]; inputs=[tuple(x[np.r_[i,j]] for x in (z,v,c,p)) for i,j in folds]
code='''
def latent_kernel(z,b,c,p,n,spec):
    a, form, warp, deg, gamma, mix = spec
    u=a*z+b; v=a*z-b
    if form==0: x=np.c_[u,v]
    elif form==1: x=np.c_[u,v,z*np.roll(z,-1,axis=1)*.20]
    elif form==2: x=np.c_[u,v,c*.25]
    elif form==3: x=np.c_[u,v,z*.25]
    else: x=np.c_[u,v,(u*v)*.5]
    x=x/max(1e-9,x[:n].std())
    if warp==1: x=np.arcsinh(x)
    if warp==2: x=x/(1+.2*x*x)**.5
    dot=x@x.T/x.shape[1]; d=np.maximum(dot.diagonal()[:,None]+dot.diagonal()-2*dot,0.)
    u=np.exp(-gamma*(x[:,None,:]-x[None,:,:])**2); m=u.mean(2); l=x.shape[1]
    pair=(l*m*m-(u*u).mean(2))/(l-1)
    return (1+dot)**deg+mix*(m+pair)+(1-mix)*np.exp(-gamma*d)

def latent_predict(z,b,c,p,target,specs):
    from scipy.linalg import cho_factor,cho_solve
    n=len(target); ans=0.
    for spec,ridge,w in specs:
        k=latent_kernel(z,b,c,p,n,spec)
        ans=ans+w*(k[n:,:n]@cho_solve(cho_factor(k[:n,:n]+ridge*np.eye(n)),target))
    return ans
'''
exec(code)
# Fit the temporal basis separately in every fold, exactly as the DAG does.
FT=[]; Y=[]
for i,j in folds:
 ss=dict(s,train={k:x[i] for k,x in t.items()}); ff=old.times(ss); FT.append(ff);Y.append(np.c_[t['y'][i],old.fit(ff,t['traj'][i].T).T])
def measure(h,ps):
 return np.array([np.abs(q-(t['y'][j] if h==0 else t['traj'][j])).mean() for q,(_,j) in zip(ps,folds)])
def loss(es): return .4*es[0]+.4*es[1:].mean()+.2*es[1:].max()
P0=[[],[]]
for xx,ff,yy,(i,j) in zip(inputs,FT,Y,folds):
 P0[0].append(a.predict_kernel(*xx,yy[:,0],a.MODELS));P0[1].append(b.predict_kernel(*xx,yy[:,1:],b.MODELS)@ff.T)
base=[measure(h,P0[h]) for h in range(2)]; print('verified baseline split MAEs',*[e.round(6).tolist() for e in base],flush=True)
records=[[],[]]
for spec in product([.10,.28,.48,.72],range(5),range(3),[1,2,3],[.03,.18,.7],[.0,.85]):
 ks=[latent_kernel(*xx,len(i),spec) for xx,(i,j) in zip(inputs,folds)]
 for ridge in [1e-6,.0002,.003]:
  pp=[[],[]]
  for k,ff,yy,(i,j) in zip(ks,FT,Y,folds):
   m=len(i); q=k[m:,:m]@cho_solve(cho_factor(k[:m,:m]+ridge*np.eye(m),check_finite=False),yy,check_finite=False); pp[0].append(q[:,0]);pp[1].append(q[:,1:]@ff.T)
  for h in range(2):
   e=measure(h,pp[h]);records[h].append((loss(e),spec,ridge,pp[h]))
for h in range(2):
 ranked=sorted(records[h],key=lambda r:r[0]); options=[]; es=base[h]
 for L in [1,3,6,12,24]:
  chosen=[]; seen=set()
  for _,spec,r,ps in ranked:
   if spec in seen:continue
   chosen.append((spec,r,ps));seen.add(spec)
   if len(chosen)==L:break
  mean=[np.mean(q,axis=0) for q in zip(*(p for _,_,p in chosen))]
  for w in [.25,.5,.75,1.]:
   ee=measure(h,[(1-w)*u+w*v for u,v in zip(P0[h],mean)])
   if ee[0]<=es[0] and ee[1:].max()<=es[1:].max() and loss(ee)<loss(es)-.001:options.append((loss(ee),w,ee,[(sp,r,1/L) for sp,r,_ in chosen]))
 if not options:continue
 _,w,ee,models=min(options,key=lambda x:x[0]);print('promote head',h,ee.round(6),flush=True)
 path=Path('/holdout_task')/('node_00.py' if h==0 else 'node_02.py');text=path.read_text(); oldline="yhat=predict_kernel(z,b,c,p,np.asarray(study['train']['y']),MODELS)" if h==0 else "weights=predict_kernel(a['z'],a['b'],a['c'],a['p'],t['modes'],MODELS)"; lhs,call=oldline.split('=',1); newcall=call.replace('predict_kernel','latent_predict').replace('MODELS','ALT_MODELS');text=text.replace(oldline,f'{lhs}={1-w!r}*{call}+{w!r}*{newcall}');path.write_text(text+'\n'+code+'\nALT_MODELS='+repr(models)+'\n')
