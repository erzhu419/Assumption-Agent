from model14 import *
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('telemetry',s['telemetry_vec']); p=s['panel_c']; u=p['u']; print('panel range',u.min(),u.max())
fix={'adj':u*np.roll(u,-1,axis=1),'opp':u*np.roll(u,2,axis=1),'cube':u**3,'square':u**2,'none':u[:,:0]}
for key in ('v1','v2'):
 v=p[key]; ans=[]
 for f in fun:
  if f in ('sq','cube'):continue
  for g,w in fix.items():
   for linear in (False,True):
    def mat(k):return D(np.c_[u if linear else u[:,:0],w,fun[f](k[0]*u)])
    def err(k):
     a=mat(k);return (a@ls(a,v)-v).ravel()
    for k0 in (0.8,2.2,3.6):
     r=least_squares(err,[k0],bounds=(.08,6),max_nfev=60); a=mat(r.x); e=r.fun.reshape(v.shape)/np.maximum(1-np.diag(a@np.linalg.pinv(a)),.001)[:,None]; ans.append((np.std(e),np.std(r.fun),f,g,linear,r.x[0],ls(a,v)))
 ans.sort(key=lambda q:q[0]); print(key)
 seen=set()
 for q in ans:
  if q[2:5] in seen:continue
  seen.add(q[2:5]); print(q[:6]);
  if len(seen)==1: print(q[-1])
  if len(seen)==10:break
