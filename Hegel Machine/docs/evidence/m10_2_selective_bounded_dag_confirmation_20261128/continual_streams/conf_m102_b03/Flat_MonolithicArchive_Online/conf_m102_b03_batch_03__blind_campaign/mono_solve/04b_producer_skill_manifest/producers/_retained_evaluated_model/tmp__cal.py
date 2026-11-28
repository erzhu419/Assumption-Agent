import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=s['panel_a']['u'],s['panel_a']['v']; rng=np.random.default_rng(2)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
lin=np.c_[np.ones(len(u)),u]; best=min([ls(lin[i],v[i]) for i in [rng.choice(len(u),9,False) for _ in range(100)]],key=lambda q:np.median(np.linalg.norm(lin@q-v,axis=1))); good=np.linalg.norm(lin@best-v,axis=1)<3; print('outliers',np.where(~good)[0]); u,v=u[good],v[good]; np.set_printoptions(precision=4,suppress=True,linewidth=160)
print('u bounds',u.min(0),u.max(0)); print('v bounds',v.min(0),v.max(0));
ph=[]; names=[]
def add(f,n): ph.append(f); names.append(n)
add(lambda z:np.ones(len(z)),'1');
for k in range(1,6):
 for ids in itertools.combinations_with_replacement(range(4),k): add(lambda z,ids=ids:z[:,ids].prod(1),''.join(str(j) for j in ids))
D=lambda z,j:np.column_stack([ph[i](z) for i in j]);
for d,m in enumerate([5,15,35,70],1):
 F=D(u,np.arange(m)); q=ls(F,v); err=v-F@q; hat=(F@np.linalg.pinv(F)).diagonal(); print('poly',d,'fit',np.sqrt(np.mean(err**2,0)),'LOO',np.sqrt(np.mean((err/np.maximum(1-hat[:,None],1e-12))**2,0))); 
 if d<=2: print(list(zip(names[:m],q)))
for i in range(4):
 for j in range(i,4):
  for sign in ([1] if i==j else [1,-1]):
   for name,f in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('ss',lambda x:x/(1+abs(x))),('rat',lambda x:1/(1+x*x)),('abs',np.abs),('sw',lambda x:x/(1+np.exp(-x)))]:
    for h in [.5,1,1.5,2,3]:
     add(lambda z,i=i,j=j,sign=sign,f=f,h=h:f(h*(z[:,i]+(sign*z[:,j] if i!=j else 0))),f'{name}({h}*{i}{sign:+d}*{j})')
F=D(u,range(len(ph))); 
for k in range(4):
 active=list(range(5));
 for turn in range(6):
  X=F[:,active]; r=v[:,k]-X@ls(X,v[:,k]); G=F-X@ls(X,F); norm=np.linalg.norm(G,axis=0); ix=np.argmax(abs(r@G)/np.maximum(norm,1e-8)); active.append(ix); print(k,turn,names[ix],np.sqrt(np.mean((v[:,k]-F[:,active]@ls(F[:,active],v[:,k]))**2)))
 print([(names[i],round(a,5)) for i,a in zip(active,ls(F[:,active],v[:,k]))])
np.savez('/tmp/caldata.npz',u=u,v=v)
