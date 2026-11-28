import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,d):
 return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(4),k)])
a=s['panel_a']; u,v=a['u'],a['v'];keep=np.linalg.norm(v,axis=1)<5; u,v=u[keep],v[keep]; print('clean rows',len(u))
for d in [1,2,3]:
 X=poly(u,d);w=fit(X,v);e=(X@w-v)/(1-np.sum(X*np.linalg.pinv(X).T,axis=1))[:,None];print('A degree',d,'loo',np.sqrt(np.mean(e**2))); names=[''.join(map(str,j)) or 'b' for k in range(d+1) for j in combinations_with_replacement(range(4),k)];print([(k,np.round(c,3).tolist()) for k,c in zip(names,w) if max(abs(c))>.025]);print('r',np.round(np.linalg.norm(X@w-v,axis=1),4))
c=s['panel_c'];u=c['u'];x=poly(u,2);print('C1',np.round(fit(x,c['v1']),3));v=c['v2']
funcs={'s':np.sin,'c':np.cos,'r':lambda x: x/(1+abs(x)),'t':np.tanh,'q':lambda x:x*x/(1+x*x)}
for m in ['s','c','ss','cc','sc','st','ct','sq','cq','sr','cr']:
 def basis(x,p):return np.column_stack([np.ones(len(x)),x]+[funcs[k](f*x) for k,f in zip(m,p)])
 def res(p):
  X=basis(u,p);return (X@fit(X,v)-v).ravel()
 starts=np.array(np.meshgrid(*([np.linspace(.4,8,8)]*len(m)))).reshape(len(m),-1).T
 p=min((least_squares(res,p,bounds=(.05,12),max_nfev=80).x for p in starts),key=lambda p:np.linalg.norm(res(p)));X=basis(u,p);r=res(p).reshape(v.shape); print('C2',m,p,np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/(1-np.sum(X*np.linalg.pinv(X).T,axis=1))[:,None])**2)))
