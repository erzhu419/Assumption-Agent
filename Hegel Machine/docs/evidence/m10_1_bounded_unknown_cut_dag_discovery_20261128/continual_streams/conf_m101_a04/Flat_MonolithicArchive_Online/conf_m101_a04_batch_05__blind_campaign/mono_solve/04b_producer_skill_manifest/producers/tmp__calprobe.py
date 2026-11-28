import pickle,itertools as it,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ['u','v']); bias=lambda x:np.c_[np.ones(len(x)),x]
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; rms=lambda a:np.sqrt(np.mean(a*a))
a=bias(u); r=v-a@fit(a,v); keep=np.linalg.norm(r,axis=1)<4*np.median(np.linalg.norm(r,axis=1)); u,v=u[keep],v[keep]; print('n',len(u),'u range',u.min(0),u.max(0)); pairs=list(it.combinations_with_replacement(range(4),2)); triples=list(it.combinations_with_replacement(range(4),3)); quad=np.column_stack([u[:,i]*u[:,j] for i,j in pairs]); cub=np.column_stack([u[:,i]*u[:,j]*u[:,k] for i,j,k in triples]); a=bias(u)
def report(name,f):
 w=fit(f,v); e=v-f@w; h=np.sum(np.linalg.qr(f)[0]**2,axis=1); print(name,'fit',rms(e),'LOO',rms(e/(1-h[:,None])),'per dim',np.round(np.sqrt(np.mean(e*e,axis=0)),6)); return w
report('affine',a); w=report('quad',np.c_[a,quad]); print('quad weights',list(zip(pairs,np.round(w[5:],4)))); report('cubic',np.c_[a,quad,cub]); report('sep powers',np.c_[a,u*u,u**3,u**4,u**5]); report('mixed quad separable cubic',np.c_[a,quad,u**3]);
sets={'u':u,'pairs+':np.column_stack([u[:,i]+u[:,j] for i,j in it.combinations(range(4),2)]),'pairs-':np.column_stack([u[:,i]-u[:,j] for i,j in it.combinations(range(4),2)]),'prod':quad[:,[1,2,3,5,6,8]]}
for sn,z in sets.items():
 for name,fun in {'sin':lambda z,w:np.sin(w*z),'cos':lambda z,w:np.cos(w*z),'sin+cos':lambda z,w:np.c_[np.sin(w*z),np.cos(w*z)],'tanh':lambda z,w:np.tanh(w*z),'rat':lambda z,w:z/(1+w*z*z)}.items():
  def mat(w):return np.c_[a,fun(z,w)]
  def err(w):f=mat(w); return rms(v-f@fit(f,v))
  xs=np.linspace(.1,10,175); j=np.argmin([err(w) for w in xs]); b=minimize_scalar(err,bounds=(xs[max(0,j-1)],xs[min(len(xs)-1,j+1)]),method='bounded'); report((sn,name,round(b.x,5)),mat(b.x))
np.set_printoptions(precision=4,suppress=True); print('inlier examples\nu',u[:12],'\nv',v[:12],'\nresiduals', (v-a@fit(a,v))[:12]);
