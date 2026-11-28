import pickle,numpy as np
from itertools import combinations_with_replacement as cr
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d):
 ids=[i for j in range(d+1) for i in cr(range(x.shape[1]),j)]; return np.column_stack([np.prod(x[:,i],axis=1) for i in ids]),[''.join(str(j) for j in i) or '1' for i in ids]
def sparse(f,y,names,base,steps=6):
 ids=list(range(base)); rest=set(range(base,len(names))); print('base',np.round(ls(f[:,ids],y),5),end='; ')
 for _ in range(steps):
  def err(i):
   a=f[:,ids+[i]];return np.std(y-a@ls(a,y))
  j=min(rest,key=err); print(names[j],round(err(j),6),end='; ');ids.append(j);rest.remove(j)
 print(' coeff',np.round(ls(f[:,ids],y),5))
if __name__=='__main__':
 for name in ['panel_a','panel_b','panel_c']:
  p=s[name]; x=p['u']; sel=np.max(np.abs(p['v']-x),axis=1)<2 if name=='panel_a' else np.ones(len(x),bool);x=x[sel];print(name,'n',len(x));f,n=poly(x,3)
  for key in p:
   if key=='u':continue
   y=p[key][sel]; print(key)
   for v in y.T:
    sparse(f,v,n,5)
    if name!='panel_a':
     for transform in [np.arctanh,np.arcsin]:
      def obj(scale):
       z=transform(v/scale)*scale;return np.std(z-f[:,:5]@ls(f[:,:5],z))
      opt=minimize_scalar(obj,bounds=(max(abs(v))*1.00001,4),method='bounded');print(transform.__name__,round(opt.x,5),round(opt.fun,6),end='; ')
     print()
 t=s['train']; x=t['x']; T=t['traj']; time=s['eval_times']; _,_,V=np.linalg.svd(T,full_matrices=False)
 for v in V[:3]:print('time singular vec',np.round(v,5))
 for v in [t['y'],T@V[0]]:
  f,n=poly(x,3);sparse(f,v,n,5,12)
