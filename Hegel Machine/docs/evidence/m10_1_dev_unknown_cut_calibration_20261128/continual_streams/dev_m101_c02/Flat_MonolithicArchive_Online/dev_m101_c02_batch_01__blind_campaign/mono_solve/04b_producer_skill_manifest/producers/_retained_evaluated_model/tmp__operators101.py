import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
L=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
I=list(combinations(range(4),2))
A=s['panel_a']; u,v=A['u'],A['v']; good=np.max(abs(u-v),axis=1)<1
co=np.array([L(np.c_[np.ones(sum(good)),u[good,j]],v[good,j]) for j in range(4)]); bias,gain=co.T
print('calibration',bias,gain)
coords={'raw':lambda u:u,'inverse':lambda u:(u-bias)/gain,'forward':lambda u:u*gain+bias}
funcs={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rational':lambda x:x/(1+x*x),'cos':lambda x:1-np.cos(x),'square':np.square,'identity':lambda x:x}
def design(u,h,kind):
    X=np.c_[np.ones(len(u)),h]
    if 'l' in kind:X=np.c_[X,u]
    if 'p' in kind:X=np.c_[X,*[u[:,i]*u[:,j] for i,j in I]]
    if 'h' in kind:X=np.c_[X,*[h[:,i]*h[:,j] for i,j in I]]
    return X
def loo(X,Y):
    q=np.linalg.svd(X,full_matrices=False)[0]; return np.mean(abs((Y-X@L(X,Y))/(1-(q*q).sum(1))[:,None]))
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2'),('train','all')]:
    p=s[pn]; U=p['x' if pn=='train' else 'u']; Y=np.c_[p['y'],p['traj'][:,-1]] if vn=='all' else p[vn]; out=[]
    for n,f in funcs.items():
     for c,transform in coords.items():
      u=transform(U)
      for k in ('','p','h','l','lp','lh') if n!='identity' else ('','p'):
       for rate in (.65,1.,1.35,1.7,2.) if n not in ('identity','square') else (1.,):
        X=design(u,f(rate*u),k); out.append((loo(X,Y),n,c,k,rate))
    print(pn,vn,'best fixed',sorted(out)[:15]);
    for _,n,c,k,r0 in sorted(out)[:5]:
      if n in ('identity','square'):continue
      u=coords[c](U); f=funcs[n]
      for d in (1,4):
       def des(r):return design(u,f(u*r),k)
       def res(r):
        X=des(r);return (Y-X@L(X,Y)).ravel()
       opt=least_squares(res,np.full(d,r0),bounds=(.15,4)); X=des(opt.x)
       print('profile',n,c,k,'rate',opt.x,'MAE',np.mean(abs(res(opt.x))),'LOO_fixed',loo(X,Y),'coeff',np.round(L(X,Y),4).tolist() if np.mean(abs(res(opt.x)))<.001 else '')
