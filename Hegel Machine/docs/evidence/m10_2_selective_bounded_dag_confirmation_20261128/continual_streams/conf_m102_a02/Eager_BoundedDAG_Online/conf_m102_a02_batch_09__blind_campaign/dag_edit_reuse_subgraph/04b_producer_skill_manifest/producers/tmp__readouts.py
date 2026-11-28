import pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def ls(a,y): return np.linalg.lstsq(a,y,rcond=1e-10)[0]
def aff(a):return np.c_[np.ones(len(a)),a]
def cal(u,v,kind='sin'):
    keep=np.linalg.norm(v-u,axis=1)<2.5
    u,v=u[keep],v[keep]
    f=np.sin if kind.startswith('sin') else lambda t:t/(1+t*t)
    blocks=lambda a,i,w:aff(np.c_[a[:,i],a[:,(i+2)%4],f(w*a[:,i]),f(w*a[:,(i+2)%4])])
    def obj(w,ret=False):
        c=np.array([ls(blocks(u,i,w),v[:,i]) for i in range(4)])
        err=v-np.column_stack([blocks(u,i,w)@c[i] for i in range(4)])
        return (c,err) if ret else np.mean(err**2)
    o=minimize_scalar(obj,bounds=(1.7,3.1) if kind.startswith('sin') else (.4,1.1),method='bounded'); C,err=obj(o.x,True)
    return o.x,C,err

def invert(x,w,C,kind):
    f=np.sin if kind.startswith('sin') else lambda t:t/(1+t*t)
    df=np.cos if kind.startswith('sin') else lambda t:(1-t*t)/(1+t*t)**2
    z=x.copy(); j=np.arange(4); k=(j+2)%4
    for _ in range(25):
        e=C[:,0]+z*C[:,1]+z[:,k]*C[:,2]+f(w*z)*C[:,3]+f(w*z[:,k])*C[:,4]-x
        J=np.zeros((len(x),4,4)); J[:,j,j]=C[:,1]+w*df(w*z)*C[:,3];J[:,j,k]=C[:,2]+w*df(w*z[:,k])*C[:,4]
        dz=np.linalg.solve(J,e[...,None])[...,0]; z-=np.clip(dz,-.5,.5)
    return z

def pol(x,d):return np.column_stack([np.ones(len(x))]+[x[:,i].prod(axis=1) for k in range(1,d+1) for i in cwr(range(x.shape[1]),k)])

def design(z):
    b=s['panel_b']; g=aff(z)@ls(aff(b['u']),b['v']);
    v=z*np.roll(g,1,axis=1); q=g[:,:2]*g[:,2:];
    return {'z':aff(z),'q':pol(z,2),'c':pol(z,3),'arc':aff(np.c_[g,z,v,q]),'sin':aff(np.c_[z,np.sin(2.15*z)]),'q+s':np.c_[pol(z,2),np.sin(2.15*z)],'q+s+zsin':np.c_[pol(z,2),np.sin(2.15*z),(z[:,:,None]*np.sin(2.15*z[:,None,:])).reshape(len(z),-1)],'q+tanh':np.c_[pol(z,2),np.tanh(2.15*z)],'q+t+s':np.c_[pol(z,2),np.tanh(2.15*z),np.sin(2.15*z)]}
if __name__=='__main__':
 t=s['train'];x=t['x']; Y=np.c_[t['y'],t['traj']];rng=np.random.default_rng(17); folds=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(12)]]
 for kind in ('sin','rat'):
    w,C,E=cal(**s['panel_a'],kind=kind);print(kind,w,'resid rmse',np.sqrt(np.mean(E**2)),C)
    z=invert(x,w,C,kind);print('inverse extents',z.min(),z.max());
    for nm,A in design(z).items():
      err=np.array([np.mean(abs(A[b]@ls(A[a],Y[a])-Y[b]),axis=0) for a,b in folds]);print(nm,'first,mean,worst',np.c_[err[:,0],err[:,1:].mean(axis=1),err[:,-1]][[0]],err[:,0].mean(),err[:,1:].mean(),np.max(err,axis=0)[[0,-1]])
      if nm in ('z','q'): print('coeff y, final',ls(A,Y)[:,[0,-1]].T)
