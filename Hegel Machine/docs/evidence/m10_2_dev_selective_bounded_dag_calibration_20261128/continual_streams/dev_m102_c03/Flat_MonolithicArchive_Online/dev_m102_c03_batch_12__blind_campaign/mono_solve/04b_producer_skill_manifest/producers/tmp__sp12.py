from laws12 import *
import glob,importlib.util
p=glob.glob('/tmp/hegel_continual_archive/*11_solver.py')[0]; sp=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
t=s['train']; x=np.vstack((t['x'],s['test']['x'])); z=m._calibrate(s['panel_a'],x)
pa=s['panel_a']; print('cal reverse',np.quantile(np.linalg.norm(m._calibrate(pa,pa['v'])-pa['u'],axis=1),[0,.5,.8,.9,1]), 'coordinates',z.min(0),z.max(0))
b=s['panel_b']; c=s['panel_c']; cy=lambda u:aff(u*np.roll(u,-1,axis=1)); co=lambda u:aff(np.cos(2.4*u)); B=aff(z)@ls(aff(b['u']),b['v']); C=cy(z)@ls(cy(c['u']),c['v1']); D=co(z)@ls(co(c['u']),c['v2']);
Z={'z':z,'b':B,'c':C,'d':D,'cos':np.cos(2.4*z),'cy':z*np.roll(z,-1,axis=1)}
from itertools import combinations
F={'1':np.ones(len(z))}
for a,f in Z.items():
 for i in range(4):F[f'{a}{i}']=f[:,i]
for a,b in [('z','z'),('c','d'),('b','c'),('b','d'),('c','c'),('d','d'),('z','d'),('z','c'),('b','b')]:
 jrange=[(i,j) for i in range(4) for j in range(4) if a!=b or i<=j];
 for i,j in jrange:F[f'{a}{i}*{b}{j}']=Z[a][:,i]*Z[b][:,j]
 F[f'{a}.{b}']=np.sum(Z[a]*Z[b],1)
for j in combinations_with_replacement(range(4),3):F['z'+''.join(map(str,j))]=np.prod(z[:,j],1)
U,S,V=np.linalg.svd(t['traj'][:52],full_matrices=False); T=np.c_[t['y'],t['traj']@V[:2].T]
def report(names):
 a=np.column_stack([F[k] for k in names]); p=a@ls(a[:52],T[:52]); return np.round(np.mean(abs(p[:52]-T[:52]),0),6),np.round(np.r_[np.mean(abs(p[52:72,0]-T[52:,0])),np.mean(abs(p[52:72,1:]@V[:2]-t['traj'][52:]))],6)
base=['1']+[f'{a}{i}' for a in 'zcd' for i in range(4)]
for pairs in [[],['c*d'],['b*c'],['b*d'],['c*c'],['d*d'],['z*d'],['z*c'],['z*z'],['c*d','b*c','b*d'],['c*d','z*z','z*d'],['c*d','b*c','b*d','z*z']]:
 extra=[k for k in F if len(k)==5 and k[1].isdigit() and k[3].isalpha() and k[0]+'*'+k[3] in pairs];print(pairs,len(extra),report(base+extra))
from scipy.linalg import qr
for j in range(3):
 keys=list(F); a=np.column_stack([F[k] for k in keys]); a-=a[:52].mean(0); norms=np.linalg.norm(a[:52],axis=0); a/=np.maximum(norms,1e-12); y=T[:,j]-T[:52,j].mean(); r=y[:52].copy(); ix=[]
 for k in range(25):
  i=np.argmax(abs(a[:52].T@r));ix.append(i);r=y[:52]-a[:52,ix]@ls(a[:52,ix],y[:52]);
  if k in [2,4,7,11,17,24]: print('OMP',j,k+1,np.mean(abs(r)),np.mean(abs(y[52:]-a[52:72,ix]@ls(a[:52,ix],y[:52]))),[keys[i] for i in ix])
