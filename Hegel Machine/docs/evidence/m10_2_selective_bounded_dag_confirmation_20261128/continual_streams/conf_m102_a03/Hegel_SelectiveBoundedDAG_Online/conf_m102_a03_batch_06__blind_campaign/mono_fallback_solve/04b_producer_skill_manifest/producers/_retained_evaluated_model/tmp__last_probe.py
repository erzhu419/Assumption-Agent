import sys, pickle
sys.path.insert(0,'/holdout_task')
import numpy as np
from batch_solver import features, fit, affine, kernelparts
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z,b,_=features(S); z,b=z[:72],b[:72];y=S['train']['y'];t=S['train']['traj']; ids=np.arange(len(y)); folds=[]
for seed in [0,10,17,25,42,8,29]:
 tr=ids[:52] if not seed else np.random.default_rng(seed).permutation(len(y))[:52]; folds.append((tr,np.setdiff1d(ids,tr)))
exec(open('/tmp/readout_last.py').read())
B,f=sine_branch(S['panel_b'],z); print('B frequency',f,flush=True)
regs=(.000002,.00002,.0002,.001,.004,.014,.045)
keys=None; allpred=[]; allerr=[]; allinner=[]
for train,test in folds:
 ix=np.r_[train,test]; K=readout_matrices(z[ix],b[ix],B[ix],f,52); n=52; pred=[]; err=[]; inner=[]; ym=y[train].mean(); Y=y[train]-ym; one=np.eye(n)
 if keys is None:keys=[(k,r) for k in K for r in regs]
 for k,r in keys:
  A=K[k]; F=cho_factor(A[:n,:n]+r*one,check_finite=False); co=cho_solve(F,Y,check_finite=False); P=ym+A[n:,:n]@co; pred.append(P); err.append(np.mean(abs(P-y[test]))); inner.append(np.mean(abs(co/cho_solve(F,one,check_finite=False).diagonal())))
 allpred.append(pred); allerr.append(err); allinner.append(inner)
allpred=np.array(allpred); allerr=np.array(allerr).T; inn=np.array(allinner).T; truth=np.array([y[test] for _,test in folds]); print('Top anchored:');
for i in np.argsort(allerr[:,0])[:10]:print(keys[i],np.round(allerr[i],4),round(inn[i].mean(),4))
print('Top rotated:');
for i in np.argsort(allerr[:,1:].mean(1))[:8]:print(keys[i],np.round(allerr[i],4),round(inn[i].mean(),4))
# Compare fixed, replicate-supported ensembles; do not fit weights to the 20-row gate.
score=allerr[:,1:].mean(1)+.2*allerr[:,1:].max(1); candidates=[]
def record(inds,weights):
 weights=np.array(weights,float);weights/=weights.sum(); e=np.mean(abs(np.einsum('fij,i->fj',allpred[:,inds,:],weights)-truth),1);candidates.append((e,np.array(inds),weights))
for g in [list(range(10)),[0,1,3],[0,1,2,3],[6,7,8,9]]:
 rank=[i for i in np.argsort(score) if keys[i][0][0] in g];
 for ns in [1,3,8,20,50]:record(rank[:ns],np.ones(ns))
anchor=np.argsort(allerr[:,0]);
for k in [1,3,8,20]:
 for weight in [.45,.7,1.]:
  ref=np.argsort(score)[:12]; record(np.r_[anchor[:k],ref],np.r_[np.full(k,weight/k),np.full(len(ref),(1-weight)/len(ref))])
passing=[r for r in candidates if r[0][0]<=.038]; selection=min(passing,key=lambda r:r[0][1:].mean()+.2*r[0][1:].max()) if passing else min(candidates,key=lambda r:.65*r[0][0]+.35*r[0][1:].mean()); e,inds,weights=selection
print('Selected:',np.round(e,6),[keys[i] for i in inds],weights,flush=True)
path='/holdout_task/batch_solver.py'; txt=open(path).read(); txt=txt.replace('from scipy.spatial.distance import cdist\n',''); txt=txt.replace('    yh=.75*adaptive(K,y)+.25*adaptive(K,y,\'loo\')','    B,f=sine_branch(study[\'panel_b\'],z)\n    yh=predict_readout(z,b,B,f,n,y)')
start=txt.index('\ndef adaptive('); end=txt.index('\ndef solve_batch(',start); txt=txt[:start]+txt[end:]
comb={}
for i,w in zip(inds,weights):
 if w: comb[keys[i]]=comb.get(keys[i],0.)+float(w)
extra=open('/tmp/readout_last.py').read(); extra=extra.replace('    ans={}\n','    ans={}\n    def add(key,value):\n        if key in selected_readout_keys: ans[key]=value\n'); extra=extra.replace('ans[j,degree,l,mix]=G1**degree+mix*np.exp(-d/l)','add((j,degree,l,mix),G1**degree+mix*np.exp(-d/l))');extra=extra.replace('ans[6+j,1,l,m]=G+m*(p+.2*p2+.02*p3)','add((6+j,1,l,m),G+m*(p+.2*p2+.02*p3))').replace('ans[6+j,2,l,m]=G**2+m*(p+p2+.2*p3)','add((6+j,2,l,m),G**2+m*(p+p2+.2*p3))')
extra+='\n\nREADOUT_MODELS = '+repr(comb)+'\nselected_readout_keys = {k for k,r in READOUT_MODELS}\n\ndef predict_readout(z,b,B,f,n,y):\n    matrices=readout_matrices(z,b,B,f,n); m=y.mean(); pred=np.zeros(len(z)-n); I=np.eye(n)\n    for (k,r),w in READOUT_MODELS.items():\n        A=matrices[k]; L=cho_factor(A[:n,:n]+r*I); pred+=w*(A[n:,:n]@cho_solve(L,y-m))\n    return m+pred\n'
open(path,'w').write(txt+'\n\n'+extra)
