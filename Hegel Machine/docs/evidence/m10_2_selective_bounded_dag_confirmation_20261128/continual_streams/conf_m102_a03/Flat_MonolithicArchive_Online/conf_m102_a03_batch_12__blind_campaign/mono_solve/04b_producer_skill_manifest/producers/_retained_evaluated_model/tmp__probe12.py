import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; traj=tr['traj']; _,_,vt=np.linalg.svd(traj[:52],full_matrices=False); t=traj@vt[:2].T
A=lambda x:np.c_[np.ones(len(x)),x]; fit=lambda x,y:lstsq(x,y)[0]
p=s['panel_a']; u=A(p['u']); v=p['v']; mask=np.ones(len(u),bool)
for _ in range(5):
 ca=fit(u[mask],v[mask]); err=np.linalg.norm(u@ca-v,axis=1);mask=err<3*np.median(err)
z=np.linalg.solve(ca[1:].T,(tr['x']-ca[0]).T).T

def check(label,f):
 b=fit(f[:52],np.c_[y,t][:52]); pred=f@b; e=np.abs(pred[52:]-np.c_[y,t][52:]); print(label,np.round(e.mean(0),4), 'state',round(np.abs(pred[52:,1:]@vt[:2]-traj[52:]).mean(),4))

def funcs(name,x,k):
 if name=='both': return np.c_[np.sin(k*x),np.cos(k*x)]
 if name=='sin': return np.sin(k*x)
 if name=='cos': return np.cos(k*x)
 if name=='tanh': return np.tanh(k*x)
 if name=='arctan': return np.arctan(k*x)
 if name=='ring': return np.c_[np.sin(k*x[:,0]*x[:,2]),np.sin(k*x[:,1]*x[:,3])]
 if name=='signed': return np.sign(x)*abs(x)**k

def panel_scan(p,j):
 u,v=p['u'],p[j]; out=[]
 for name in ['sin','cos','both','tanh','arctan','signed','ring']:
  for lin in [False,True]:
   def des(x,k):return A(np.c_[x,funcs(name,x,k)] if lin else funcs(name,x,k))
   def loss(k):
    d=des(u,k); return np.mean((d@fit(d,v)-v)**2)
   kg=np.linspace(.2,25,250); ll=np.array([loss(k) for k in kg]); i=ll.argmin(); k=minimize_scalar(loss,bounds=(kg[max(0,i-1)],kg[min(len(kg)-1,i+1)]),method='bounded').x
   b=fit(des(u,k),v); out.append((loss(k),name,lin,k,des(z,k)@b))
 out.sort(key=lambda e:e[0]); print(j,[(round(a,8),b,c,round(d,4)) for a,b,c,d,_ in out[:6]])
 return out

def main():
 check('z',A(z));
 for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  print(k); out=panel_scan(s[k],j)
  for _,name,lin,k,w in out[:3]:
   check((name,lin),A(w));check((name,lin,'quad'),A(np.c_[w,np.stack([w[:,i]*w[:,j] for i in range(4) for j in range(i,4)],1)]))
 for yy in (y,t[:,0],t[:,1]):
  print('univariate',[(round(v,3),i,n,round(k,2)) for v,i,n,k in sorted([(abs(np.corrcoef(funcs(n,z[:52,i],k),yy[:52])[0,1]),i,n,k) for i in range(4) for n in ('sin','cos') for k in np.linspace(.2,25,250)],reverse=True)[:9]])
if __name__=='__main__':main()
