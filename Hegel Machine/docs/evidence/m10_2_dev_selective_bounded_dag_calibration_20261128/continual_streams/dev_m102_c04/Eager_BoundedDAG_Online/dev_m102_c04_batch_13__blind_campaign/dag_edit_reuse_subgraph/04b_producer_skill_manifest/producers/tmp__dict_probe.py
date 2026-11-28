import pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_c']['u']; v=s['panel_c']['v2']; a=np.c_[np.ones(len(x)),x]; H=np.eye(len(x))-a@np.linalg.pinv(a); e=H@v; U=np.linalg.svd(e,full_matrices=False)[0]; P=U@U.T
vec=[p for p in itertools.product([0,1,-1],repeat=4) if any(p) and next(t for t in p if t)!=-1]; L=x@np.array(vec).T; funcs={}; results=[]
def add(n,fn):
 funcs[n]=fn
 def loss(k):
  y=H@fn(k); return np.linalg.norm(y-P@y)/max(np.linalg.norm(y),1e-12) if np.linalg.norm(y)>1e-9 else 1.
 grid=[0,.06,.16,.3,.6,1.2,2.15,4,8]; z=[loss(k) for k in grid]; j=np.argmin(z); o=minimize_scalar(loss,bounds=(grid[max(0,j-1)],grid[min(j+1,len(grid)-1)]),method='bounded'); results.append((o.fun,o.x,n))
for i,t in enumerate(L.T):
 for name,fn in [('sq',lambda t,k:t*t/(1+k*t*t)),('linear',lambda t,k:t/(1+k*t*t)),('abs',lambda t,k:t/(1+k*abs(t))),('tanh',lambda t,k:np.tanh(k*t)),('sin',lambda t,k:np.sin(k*t)),('cos',lambda t,k:np.cos(k*t)),('soft square',lambda t,k:t*t/(1+k*abs(t)))]:add((name,vec[i]),lambda k,t=t,fn=fn:fn(t,k))
for i,j in itertools.combinations_with_replacement(range(4),2):
 w=x[:,i]*x[:,j]
 D=[('self',w*w),('abs self',abs(w)),('norm',np.sum(x*x,axis=1)),('pair norm',x[:,i]**2+x[:,j]**2)]+[(str(lbl),t*t) for lbl,t in zip(vec,L.T)]
 for name,d in D:add(('prod',i,j,name),lambda k,w=w,d=d:w/(1+k*d))
for i,j,h in itertools.combinations(range(4),3):
 w=x[:,i]*x[:,j]*x[:,h];
 for lbl,t in zip(vec,L.T):add(('triple',i,j,h,lbl),lambda k,w=w,t=t:w/(1+k*t*t))
results.sort(); print(*results[:28],sep='\n')
f=a.copy(); xs=results[:180]; T=np.column_stack([funcs[n](k) for _,k,n in xs]); labels=[]
for it in range(8):
 H=np.eye(len(x))-f@np.linalg.pinv(f); t=H@T; gain=np.sum((t.T@(H@v))**2,axis=1)/np.maximum(1e-10,np.sum(t*t,axis=0)); j=np.argmax(gain); f=np.c_[f,T[:,j]]; labels.append(xs[j]); print('greedy',xs[j],np.sqrt(np.mean((v-f@np.linalg.lstsq(f,v,rcond=None)[0])**2)))
print('linear fit',np.linalg.lstsq(a,v,rcond=None)[0]); print('quadratic fit',np.linalg.lstsq(np.c_[a,np.array([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]).T],v,rcond=None)[0][5:])
