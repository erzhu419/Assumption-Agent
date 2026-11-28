import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
fun={'tanh':lambda u,k:np.tanh(k*u),'sin':lambda u,k:np.sin(k*u),'soft':lambda u,k:u/(1+k*abs(u)),'rat':lambda u,k:u/(1+k*u*u),'root':lambda u,k:u/np.sqrt(1+k*u*u),'sqrat':lambda u,k:u*u/(1+k*u*u),'cos':lambda u,k:np.cos(k*u),'expsq':lambda u,k:-np.expm1(-k*u*u)}
def panel(u,v,f,affine,fitids):
 def A(x,k):return np.column_stack((np.ones(len(x)),x,f(x,k))) if affine else np.column_stack((np.ones(len(x)),f(x,k)))
 def err(k):
  a=A(u[fitids],k); return np.linalg.norm(a@ls(a,v[fitids])-v[fitids])**2
 g=np.geomspace(.025,5,40); i=np.argmin([err(k) for k in g]); k=minimize_scalar(err,bounds=(g[max(0,i-1)],g[min(i+1,39)]),method='bounded').x
 w=ls(A(u[fitids],k),v[fitids]); return k,lambda x:A(x,k)@w,w
if __name__=='__main__':
 for pn,vs in [('a',['v']),('c',['v1','v2'])]:
  p=s['panel_'+pn]; u=p['u']
  for vk in vs:
   v=p[vk]; ids=np.where(np.linalg.norm(v-u,axis=1)<4)[0] if pn=='a' else np.arange(len(u)); arr=[]
   for n,f in fun.items():
    for aff in [False,True]:
     k,pred,w=panel(u,v,f,aff,ids); cv=[]
     for j in range(5):
      tr=np.setdiff1d(ids,ids[j::5]); te=ids[j::5]; _,pr,_=panel(u,v,f,aff,tr);cv.extend(abs(pr(u[te])-v[te]).ravel())
     arr.append((np.mean(cv),np.mean(abs(pred(u[ids])-v[ids])),n,aff,k,w))
   for row in sorted(arr,key=lambda p:p[0])[:6]:print(pn,vk,row[:5]);print(np.round(row[-1],5)) if row==min(arr,key=lambda p:p[0]) else None
 t=s['eval_times']; z=s['train']['traj'];
 for f in [lambda p:-np.expm1(-t[:,None]*p),lambda p:np.c_[-np.expm1(-p[0]*t),t*np.exp(-p[1]*t)],lambda p:np.c_[-np.expm1(-p[0]*t),t*np.exp(-p[1]*t)/(1+.22*t)]]:
  def re(p):
   a=f(p);return (a@ls(a,z[:52].T)-z[:52].T).ravel()
  p=least_squares(re,[2,.3],bounds=(.001,10)).x;print('time',p,np.mean(abs(re(p))),np.mean(abs(f(p)@ls(f(p),z[52:].T)-z[52:].T)))
