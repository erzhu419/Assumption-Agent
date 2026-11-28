import pickle, numpy as np
from model11 import *
from itertools import combinations_with_replacement, combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train'];z=calibrate(s['panel_a'],tr['x']); b,(p,pk,pe),(q,qk,qe)=panels(s,z); h,amp,r,e=trajectory(s['eval_times'],tr['traj'][:52]); amp=ls(h,tr['traj'].T).T; y=np.c_[tr['y'],amp]; print('z range',z.min(0),z.max(0),'operators',pk,pe,qk,qe,'rates',r,e,'amp std',amp.std(0))
blocks={'z':z,'b':b,'p':p,'q':q}; funcs={'tanh':lambda a:np.tanh(a),'sin':lambda a:np.sin(2.15*a),'s1':np.sin,'cos':lambda a:np.cos(2.15*a),'rat':lambda a:a/(1+.22*a*a),'soft':lambda a:a/(1+.84*abs(a)),'sq':lambda a:a*a,'cub':lambda a:a**3,'log':lambda a:np.logaddexp(0,2.2*a)}
for name,x in list(blocks.items()):
 for fn,f in funcs.items():blocks[fn+name]=f(x)
for (na,a),(nb,b) in combinations_with_replacement(list(blocks.items())[:4],2):
 for k in range(4):
  if na==nb and k==0:continue
  ab=a*np.roll(b,k,axis=1);blocks[f'{na}*{nb}{k}']=ab;blocks[f'dot{na}{nb}{k}']=ab.sum(1)[:,None]

def score(names,ret=False):
 x=D(np.column_stack([blocks[k] for k in names])); c=ls(x[:52],y[:52]); pred=x@c; e=abs(pred-y); return e[:52].mean(0),e[52:].mean(0),abs((pred[52:,1:]-y[52:,1:])@h.T).mean()
for i in range(3):
 pool=[(score([k])[0][i],k) for k in blocks]; print('best single',i,[(k,np.round(score([k])[1],4)) for _,k in sorted(pool)[:10]])
for i in range(3):
 stages=[(0,())]
 for depth in range(1,6):
  cand={tuple(sorted((*a,k))) for _,a in stages for k in blocks if k not in a}; stages=sorted((score(a)[0][i]+.0007*sum(blocks[k].shape[1] for k in a),a) for a in cand)[:14]
  print(i,depth,[(a,np.round(score(a)[0][i],5),np.round(score(a)[1][i],5),round(score(a)[2],5)) for _,a in stages[:4]])
print('linear combos');
for a in [('z','b','p','q'),('b','p','q'),('p','q'),('z','p','q')]:print(a,score(a))
