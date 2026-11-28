import pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task')
import node_00 as m
from _frozen_nodes import node_01
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); ix=np.arange(n)
x=m.features(s,tr['x']); stats={}; family_stats={}
for it in (ix[52:],ix[:20],ix[20:40],ix[40:60]):
 ir=np.setdiff1d(ix,it); fi=[a[np.r_[ir,it]] for a in x]; ts={**s,'train':{**tr,'traj':tr['traj'][ir]}}; tp=node_01.run_node({'__input__':ts}); y=tr['y']; t=tr['traj']; truth=np.c_[y[it],t[it]]; tar=np.c_[y[ir],t[ir]]; p=np.c_[y[ir],tp['modes']@tp['basis'].T]; rows=list(m.operators(*fi,tar)); folds={}
 def error(w):
  e=np.abs(w@p-truth).mean(0); return np.r_[e[0],e[1:].mean()]
 for k,score,w in rows:stats.setdefault(k,[]).append(error(w))
 for family in ['all','b','bd','zbd','mixed','sym','sparse','b,bd,zbd','bd,zbd,mixed,sym']:
  for j in range(2):
   cand=[row for row in rows if family=='all' or row[0][0][0].rstrip('+') in family.split(',')]; rank=sorted(cand,key=lambda r:r[1][0] if j==0 else np.mean(r[1][1:]/t[ir].std(0)))[:9]; val=error(np.mean([r[2] for r in rank],axis=0))[j]; family_stats.setdefault((family,j),[]).append(val); folds[family,j]=round(val,5)
 print('slice',it[0],folds)
for j,head in enumerate('YT'):
 best_family=min((v for v in family_stats if v[1]==j),key=lambda v:max(family_stats[v])+np.mean(family_stats[v])); fs=family_stats[best_family]; print('best nested family',head,best_family[0],np.round(fs,5)); best=sorted(stats,key=lambda k:np.max(np.array(stats[k])[:,j])+np.mean(np.array(stats[k])[:,j]))[:5];print('fixed diagnostic',[(k,np.round(np.array(stats[k])[:,j],5).tolist()) for k in best]);
 # Promote the best training-PRESS-selected family across the four external slices.
 path='/holdout_task/node_00.py'; a=open(path).read(); a=a.replace(f"FAMILY_{head} = 'all'",f'FAMILY_{head} = {best_family[0]!r}');open(path,'w').write(a)
