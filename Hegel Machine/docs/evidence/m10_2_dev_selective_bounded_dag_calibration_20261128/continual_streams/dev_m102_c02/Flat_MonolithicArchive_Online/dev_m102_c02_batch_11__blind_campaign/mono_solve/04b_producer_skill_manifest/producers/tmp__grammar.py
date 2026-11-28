import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; A=np.load('/tmp/cached.npz'); z=A['z']; y=s['train']['y']; n=52
ls=lambda h,v:np.linalg.lstsq(h,v,rcond=1e-10)[0]
a=lambda x:np.c_[np.ones(len(x)),x]
ops={'soft':(lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[2.15,0],([.02,-6],[12,6])),'dead':(lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.22,1.25],([0,.25],[.8,4])),'atan':(lambda x,p:np.arctan(p[0]*x),[2.15],([.01],[12])),'sin':(lambda x,p:np.sin(p[0]*x+p[1]),[2.15,0],([.01,-4],[12,4])),'rational':(lambda x,p:x/(1+p[0]*x*x),[.22],([0],[6])),'sat':(lambda x,p:x/(1+p[0]*x*x)**.5,[.22],([0],[6])),'exp':(lambda x,p:np.exp(np.clip(p[0]*x,-30,30)),[.7],([-8],[8])),'relu':(lambda x,p:np.maximum(x-p[0],0),[.22],([-1.5],[1.5]))}
D={}
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[pn]['u'],s[pn][vn]; fits=[]
 for key,(fn,p,bd) in ops.items():
  for lin in [False,True]:
   for mix in [0,1,2]:
    i,j=np.triu_indices(4)
    def H(x,p):
     q=fn(x,p); cols=[q];
     if lin:cols.append(x)
     if mix==1:cols.append(x*np.roll(x,1,axis=1))
     if mix==2:cols.append(x[:,i]*x[:,j])
     return a(np.column_stack(cols))
    def res(p):
     h=H(u,p); return (h@ls(h,v)-v).ravel()
    opt=least_squares(res,p,bounds=bd,max_nfev=120); h=H(u,opt.x); q=H(z,opt.x)@ls(h,v)
    S=np.linalg.svd(h,full_matrices=False)[0][:,:np.linalg.matrix_rank(h)]; err=abs(res(opt.x).reshape(v.shape)); cv=np.mean(err/np.maximum(1-(S*S).sum(1,keepdims=True),.01)); fits.append((cv,key,lin,mix,opt.x,q))
 fits.sort(key=lambda r:r[0]); print(pn,vn,[(round(r[0],7),*r[1:-1]) for r in fits[:8]])
 for cv,key,lin,mix,p,q in fits[:5]: D[pn+vn+key+str(lin)+str(mix)]=q
np.savez('/tmp/grammar.npz',**D)
