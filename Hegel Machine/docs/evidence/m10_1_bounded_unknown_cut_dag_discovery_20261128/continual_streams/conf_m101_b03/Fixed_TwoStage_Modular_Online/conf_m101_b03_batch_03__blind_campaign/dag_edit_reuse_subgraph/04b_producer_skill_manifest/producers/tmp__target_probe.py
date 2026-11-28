import itertools as it,pickle,runpy,numpy as np
P='/holdout_task/'; batch=pickle.load(open(P+'public_batch.pkl','rb')); n=runpy.run_path(P+'node_00.py'); sink=runpy.run_path(P+'_frozen_nodes/node_01.py')
for item in batch:
 s=item['study']; tr=s['train']; p=s['panel_a']; c=n['_fit_panel_a'](p['u'],p['v']); x=np.vstack([tr['x'],s['test']['x']]); z=n['_panel_a'](p,x); print('study',list(s),'item',list(item),'panels',[(k,list(s[k])) for k in ('panel_a','panel_b','panel_c')]); print('A coefficients',c.round(5),'inverse max residual',abs(n['_a_forward'](z,c)-x).max(),'z ranges',z.min(0),z.max(0)); p=s['panel_c']; d=np.c_[np.ones(len(p['u'])),np.cos(2.4*p['u'])]; b=np.linalg.lstsq(d,p['v2'],rcond=None)[0]; print('C fit',abs(d@b-p['v2']).max(),'mixing',b.round(4)); z=z[:len(tr['x'])]; h,m=sink['_modes'](s['eval_times'],tr['traj']); print('trajectory mode basis error',abs(m@h.T-tr['traj']).max()); Y=np.c_[tr['y'],m]; D=[]; names=[]
 def add(name,a):
  D.append(a);names.append(name)
 def terms(name,a):
  add(name,a)
  for k in (.5,.7,1.,1.2,1.5,1.7,2.,2.2,2.4,2.5,2.8,3.,3.5,4.):
   for fn in (np.sin,np.cos,np.tanh):add(f'{fn.__name__}({k} {name})',fn(k*a))
   add(f'rat({k} {name})',a/(1+k*a*a));add(f'inv({k} {name})',1/(1+k*a*a))
 for i,a in enumerate(z.T):
  terms(str(i),a)
  for k in (2,3,4):add(f'{i}^{k}',a**k)
 for i,j in it.combinations(range(4),2):
  a,b=z[:,i],z[:,j]; terms(f'{i}*{j}',a*b); terms(f'{i}+{j}',a+b); terms(f'{i}-{j}',a-b)
  for k in (.7,1.,1.5,2.,2.4,2.5,3.):
   for fn in (np.sin,np.cos,np.tanh):
    add(f'{fn.__name__}({k} {i})*{j}',fn(k*a)*b);add(f'{fn.__name__}({k} {j})*{i}',fn(k*b)*a)
 for k in (3,4):
  for ids in it.combinations_with_replacement(range(4),k):add(str(ids),np.prod(z[:,ids],1))
 for key in ('panel_b','panel_c'):
  q=s[key]; u=q['u']; d=np.c_[np.ones(len(u)),u]
  for vname,v in q.items():
   if vname=='u':continue
   for form,f in [('aff',lambda u:u),('cos',lambda u:np.cos(2.4*u))]:
    b=np.linalg.lstsq(np.c_[np.ones(len(u)),f(u)],v,rcond=None)[0];print(key,vname,form,np.sqrt(np.mean((np.c_[np.ones(len(u)),f(u)]@b-v)**2)))
    for j,vv in enumerate((np.c_[np.ones(len(z)),f(z)]@b).T):add(f'{key} {vname} {form} {j}',vv)
 D=np.array(D).T; D=(D-D.mean(0))/D.std(0)
 for y in Y.T:
  y=y-y.mean(); ids=[]; r=y.copy()
  for step in range(9):
   Q=np.linalg.qr(D[:,ids],mode='reduced')[0]; E=D-Q@(Q.T@D); score=abs(E.T@r)/np.maximum(np.linalg.norm(E,axis=0),1e-8); score[ids]=-1; ids.append(score.argmax()); A=D[:,ids]; coef=np.linalg.lstsq(A,y,rcond=None)[0]; r=y-A@coef; print(step+1,names[ids[-1]],round(np.mean(abs(r)),5),round(np.mean(abs(r)/np.maximum(1-1/len(y)-np.diag(A@np.linalg.pinv(A)),.01)),5),coef.round(3))
