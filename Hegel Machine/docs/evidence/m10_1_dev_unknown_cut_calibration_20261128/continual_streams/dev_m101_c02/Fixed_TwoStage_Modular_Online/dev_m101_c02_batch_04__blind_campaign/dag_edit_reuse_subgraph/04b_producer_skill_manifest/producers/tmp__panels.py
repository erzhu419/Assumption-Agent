import pickle,numpy as np
from itertools import combinations_with_replacement
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=3,suppress=True,linewidth=135)
def feats(x, degree=2, omega=None):
    a=[np.ones(len(x))]; names=['1']
    for d in range(1,degree+1):
        for i in combinations_with_replacement(range(4),d):
            a.append(x[:,i].prod(axis=1)); names.append(''.join(str(j) for j in i))
    if omega is not None:
        a.extend([*np.sin(omega*x.T),*np.cos(omega*x.T)]); names.extend([f'{f}{i}' for f in ('s','c') for i in range(4)])
    return np.stack(a,axis=1),names
def fit(X,y,robust=False):
    w=np.ones(len(y))
    for i in range(45 if robust else 1):
        b=np.linalg.lstsq(X*w[:,None],y*w[:,None],rcond=None)[0]
        r=np.linalg.norm(X@b-y,axis=1)
        w=np.power(1+(r/max(.002,.7*np.median(r)))**2,-.5)
    return b,r,w
if __name__=='__main__':
 for k in ('panel_a','panel_b','panel_c','train','test'):
    for key,x in p[k].items():
        if x.ndim==2 and x.shape[1]==4:print(k,key,'mean',x.mean(0),'sd',x.std(0),'range',x.min(0),x.max(0),'sample',x[:2])
    if k not in ('panel_a','panel_b','panel_c'):continue
    for key in p[k]:
      if key=='u':continue
      for src,dest in [('u',key),(key,'u')]:
        print(k,src,'->',dest)
        x,y=p[k][src],p[k][dest]
        for deg,om in [(1,None),(2,None),(1,1.),(2,1.),(2,1.5),(2,2.)]:
          X,n=feats(x,deg,om); b,r,w=fit(X,y,k=='panel_a');
          H=(X*np.linalg.pinv(X).T).sum(1)
          print(deg,om,'resid',np.quantile(r,[0,.5,.9,1]),'LOO (OLS)' if k!='panel_a' else 'inliers',np.median(r/(1-H)) if k!='panel_a' else np.sum(w>.5),'cond',round(np.linalg.cond(X)))
          if deg==2 and om is None and src=='u':print(np.array(n),b.T)
