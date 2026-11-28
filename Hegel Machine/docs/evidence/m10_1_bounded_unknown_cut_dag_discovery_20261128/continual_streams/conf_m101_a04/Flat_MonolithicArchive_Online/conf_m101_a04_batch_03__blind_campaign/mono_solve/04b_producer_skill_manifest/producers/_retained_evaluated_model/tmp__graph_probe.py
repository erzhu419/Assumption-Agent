import pickle,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def calibrate(panel,raw):
 u,v=panel['u'],panel['v']; nxt=np.roll(u,-1,axis=1); keep=np.ones(len(u),bool)
 A=np.c_[np.ones(len(u)),u]
 for _ in range(8):
  r=np.linalg.norm(A@fit(A[keep],v[keep])-v,axis=1); keep=r<5*np.median(r)
 def design(x):
  return np.stack((np.ones_like(x),x,x**3,(x-1)*np.roll(x,-1,axis=1)),axis=2)
 D=design(u)
 for _ in range(5):
  w=np.stack([fit(D[keep,j],v[keep,j]) for j in range(4)]); r=np.linalg.norm((D*w).sum(2)-v,axis=1);keep=r<max(.0006,5*np.median(r))
 print('cal',keep.sum(),w,'RMS',np.std(r[keep]))
 z=(raw-w[:,0])/w[:,1]; j=np.arange(4)
 for _ in range(35):
  pred=(design(z)*w).sum(2); J=np.zeros((len(z),4,4));J[:,j,j]=w[:,1]+3*w[:,2]*z*z+w[:,3]*np.roll(z,-1,axis=1); J[:,j,(j+1)%4]=w[:,3]*(z-1); delta=np.linalg.solve(J,(pred-raw)[...,None])[...,0]; z-=np.clip(delta,-.4,.4)
 print('inverse max',np.max(abs((design(z)*w).sum(2)-raw)), 'ranges',z.min(0),z.max(0));return z
def op(u,v,x,name):
 def design(x,f):
  h=(np.sin(f*x),np.sin(2*f*x)) if name=='sin' else (np.cos(f*x),)
  return np.column_stack((np.ones(len(x)),x,*h))
 def residual(f):
  a=design(u,f);return (a@fit(a,v)-v).ravel()
 f0=3.285 if name=='sin' else 1.26
 for mode in [1,4]:
  f=least_squares(residual,np.full(mode,f0),bounds=(f0*.75,f0*1.25),gtol=1e-13).x
  print(name,mode,f,np.std(residual(f)))
 a=design(u,f);return design(x,f)@fit(a,v)
def poly(x,k):return np.column_stack([x[:,p].prod(1) for i in range(k+1) for p in itertools.combinations_with_replacement(range(x.shape[1]),i)])
def main():
 t=s['train']; raw=np.r_[t['x'],s['test']['x']]; x=calibrate(s['panel_a'],raw); p=s['panel_b']; b=np.c_[np.ones(len(x)),x]@fit(np.c_[np.ones(len(p['u'])),p['u']],p['v']); p=s['panel_c']; c=op(p['u'],p['v1'],x,'sin'); d=op(p['u'],p['v2'],x,'cos'); np.savez('/tmp/branches.npz',x=x,b=b,c=c,d=d,raw=raw,y=t['y'],T=t['traj']); Y=np.c_[t['y'],t['traj']]
 print('TRAIN/CV scalar and trajectory'); n=52
 def check(A,label,l=0):
  tr=A[:n]; w=fit(tr,Y[:n]) if l==0 else np.linalg.solve(tr.T@tr+l*np.eye(tr.shape[1]),tr.T@Y[:n]); E=abs(A[:72]@w-Y); print(label,A.shape[1], np.round([E[:n,0].mean(),E[n:,0].mean(),E[:n,1:].mean(),E[n:,1:].mean()],5))
 for name,z in [('raw',raw),('x',x),('B',b),('C',c),('D',d),('CD',np.c_[c,d]),('XCD',np.c_[x,c,d])]:
  for k in (1,2,3):
   if k==3 and z.shape[1]>4:continue
   check(poly(z,k),name+str(k),.0003 if z.shape[1]>4 and k==2 else 0)
 for nm,u,v in [('CD',c,d),('BC',b,c),('BD',b,d),('XC',x,c),('XD',x,d)]:
  uv=u*v; base=np.c_[np.ones(len(x)),u,v];check(np.c_[base,uv.sum(1)],nm+' matched');check(np.c_[base,uv],nm+' diag');check(np.c_[base,(u[:,:,None]*v[:,None,:]).reshape(-1,16)],nm+' cross')
if __name__=='__main__':main()
