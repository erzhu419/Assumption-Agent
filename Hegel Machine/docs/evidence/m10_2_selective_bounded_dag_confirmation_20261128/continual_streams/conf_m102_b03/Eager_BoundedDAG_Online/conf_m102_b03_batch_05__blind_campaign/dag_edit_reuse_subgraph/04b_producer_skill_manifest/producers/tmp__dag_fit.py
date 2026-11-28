from dag_probe import *
from itertools import combinations_with_replacement
rng=np.random.default_rng(91)
def terms(d): return [()]+[c for k in range(1,d+1) for c in combinations_with_replacement(range(4),k)]
def Aof(u,tt): return np.column_stack([np.prod(u[:,c],axis=1) for c in tt])
def cal(p):
 u,v=p['u'],p['v']; tt=terms(3); A=Aof(u,tt); B=[]
 for j in range(4):
  y=v[:,j]; cy=(j+1)%4; h0=np.abs(y-ls(ones(u),y)@ones(u).T)<2.5
  # Gross corruptions are orders of magnitude larger than calibration noise.
  h=h0.copy(); b=ls(A[h],y[h]); print('cal initial',j,sum(h),np.quantile(abs(A[h]@b-y[h]),[.5,1]),[(c,round(w,6)) for c,w in zip(tt,b) if abs(w)>.001]);
  for thresh in [.01,.003,.001]:
   keep=abs(b)>thresh; b2=np.zeros(len(tt)); b2[keep]=ls(A[h][:,keep],y[h]);
   if np.max(abs(A[h]@b2-y[h]))<.001:b=b2
  B.append(b)
 return tt,np.array(B).T
def inv(x,tt,B):
 from scipy.optimize import root
 z0=(x-B[0])@np.linalg.inv(B[1:5]);return np.array([root(lambda z: Aof(z[None],tt)[0]@B-y,z).x for z,y in zip(z0,x)])
def fitness(F,y,key,splits):
 o=[]
 for i,j in splits:
  p=F[j]@ls(F[i],y[i]);o.append(np.mean(abs(p-y[j])))
 return key,F.shape[1],np.round(o,5)
if __name__=='__main__':
 tt,B=cal(s['panel_a']); p=s['panel_c']; u=p['u']; train=s['train']; z=inv(train['x'],tt,B); print('inverse',np.max(abs(Aof(z,tt)@B-train['x'])),np.max(abs(z)),z[:2]);
 A=poly(u,2); bw=ls(A,p['v1']);print('branch1 coefficients',[(c,np.round(w,5).tolist()) for c,w in zip(terms(2),bw)]);c=poly(z,2)@bw; d=old._branch(p,z,2);
 print('branch2 amplitudes range', np.min(d,axis=0),np.max(d,axis=0)); q=s['panel_b']; bw=ls(ones(q['u']),q['v']); print('panelB',np.round(bw,6)); zm=ones(z)@bw;
 dot=np.sum(c*d,axis=1); cross=(d[:,:,None]*c[:,None,:]).reshape(len(z),-1); zz=np.c_[c,d]; print('telemetry',s['telemetry_vec']);
 Fs={'arch scalar':ones(np.c_[zz,dot]),'arch traj':ones(np.c_[zz,cross]),'arch+z':ones(np.c_[z,zz,cross]),'polyz2':poly(z,2),'polyz3':poly(z,3),'polyc+d2':poly(zz,2),'diag c+d':ones(np.c_[zz,c*d,zz**2]),'atan c+d':ones(np.c_[zz,np.arctan(1.25*zz),cross]),'tanh c+d':ones(np.c_[zz,np.tanh(1.25*zz),cross]),'cubic c+d':ones(np.c_[zz,zz**2,zz**3,cross]),'cubic+dot':ones(np.c_[zz,zz**2,zz**3,dot]),'sin c+d':ones(np.c_[zz,np.sin(1.25*zz),cross]),'rat c+d':ones(np.c_[zz,zz/(1+1.25*zz**2),cross]),'softsign c+d':ones(np.c_[zz,zz/(1+1.25*abs(zz)),cross])};
 ids=np.arange(len(z)); splits=[(ids[:52],ids[52:])]; splits.extend((np.setdiff1d(ids,j),j) for j in np.array_split(rng.permutation(ids),4));
 for k,f in Fs.items(): print(fitness(f,train['y'],k,splits),fitness(f,train['traj'],k,splits)[2]);
 np.savez('/tmp/dag_features.npz',z=z,c=c,d=d,zm=zm,y=train['y'],t=train['traj'])
