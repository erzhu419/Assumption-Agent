import sys, pickle, importlib
import numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
a,b,c=[importlib.import_module('node_%02d'%i).run_node for i in range(3)]
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=st['train']; n=len(t['y']); rng=np.random.default_rng(604)
splits=[np.arange(n-20,n)]
for _ in range(5): splits.extend(np.array_split(rng.permutation(n),4))
order=np.argsort(t['y']); splits.extend([order[:18], order[-18:]])
P=[]; S=[]; T=[]; Y=[]; J=[]; I=[]
for k,ix in enumerate(splits):
    tr=np.setdiff1d(np.arange(n),ix); sh=a({'__input__':dict(st,train={q:t[q][tr] for q in ('x','y','traj')},test={'x':t['x'][ix]})}); po=b({'node_00':sh})['poly']; so=c({'node_00':sh}); sm=so['smooth']; pred=(.4*po[:,1:]+.6*sm[:,1:])@so['vt']; P.extend(po[:,0]); S.extend(sm[:,0]); Y.extend(t['y'][ix]); T.extend(np.mean(abs(pred-t['traj'][ix]),axis=1)); J.extend([k]*len(ix)); I.extend(ix)
P,S,Y,T,J,I=map(np.asarray,(P,S,Y,T,J,I)); np.savez('/tmp/sink_eval.npz',P=P,S=S,Y=Y,T=T,J=J,I=I)
for w in [-1,-.5,-.25,0,.15,.25,.35,.5,.65,.8,1,1.25,1.5,2]:
 e=abs(S+w*(P-S)-Y); f=np.array([e[J==k].mean() for k in range(len(splits))]); print(f'w_poly {w:5.2f} public {f[0]:.6f} random avg/worst {f[1:21].mean():.6f}/{max(f[1:21]):.6f} response tails {f[-2:]}')
print('trajectory public/ordinary worst/tails',T[J==0].mean(),max(T[J==j].mean() for j in range(1,21)),[T[J==j].mean() for j in [21,22]])
print('public residual mean, poly/smooth',np.mean((P-Y)[J==0]),np.mean((S-Y)[J==0])); print('public cases cols y poly smooth delta'); print(np.round(np.c_[Y,P,S,P-S][J==0],4))
