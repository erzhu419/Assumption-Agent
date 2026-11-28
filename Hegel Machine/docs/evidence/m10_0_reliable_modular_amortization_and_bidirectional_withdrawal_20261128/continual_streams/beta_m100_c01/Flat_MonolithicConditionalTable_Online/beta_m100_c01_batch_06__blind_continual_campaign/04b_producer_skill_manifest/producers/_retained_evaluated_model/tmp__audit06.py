import importlib.util, itertools, json, pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open(p+'public_batch.pkl','rb'))[0]; s=item['study']; y=m._calibrate(s); Z=s['Z_nuisance']; I=np.eye(Z.shape[1]); R=s['source_ridge']; G=s['nuisance_ridge']; cap=s['u_box']; budget=s['total_budget']
nx,ny=s['nx'],s['ny']; shifts=s['mu_shifts']; ij=s['sensor_ij']; slots=ij[:,0]+nx*ij[:,1]; ns=len(ij); coords=np.tile(ij,(len(shifts),1)); ids=np.arange(len(y)); rhs=np.column_stack([v.ravel(order='F') for v in s['source_modes']]); n=rhs.shape[1]; E=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
base=[]; robin=[]
for a,k in zip('xy',(nx,ny)):
 h=s['L'+a]/(k-1); base.append(m._reflecting_axis(s['kappa_'+a],s['vel_'+a],h)); robin.append(s['robin_'+a]/h)
def transport(w):
 axes=[]
 for k,A,v,(l,r) in zip((nx,ny),base,robin,(w[:2],w[2:])):
  d=np.zeros(k); d[[0,-1]]=np.array([l,r])*v; axes.append(A+diags(d,format='csc'))
 L=kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc')+V
 return np.vstack([splu(L+mu*E).solve(rhs)[slots] for mu in shifts])
groups={'sensor':[np.all(coords==q,axis=1) for q in np.unique(ij,axis=0)],'shift':[ids//ns==k for k in range(len(shifts))],'rosette':[ids%ns//4==k for k in range(ns//4)]}
def score(S,split,source_ridge=R,nuisance_ridge=G):
 pred=np.zeros_like(y); worst=0.
 for q in split:
  z=Z[~q]; X=np.column_stack([S[~q],y[~q]]); zz=z.T@z+nuisance_ridge*I; P=X-z@np.linalg.solve(zz,z.T@X); A,r=P[:,:-1],P[:,-1]; w,_=m._box_simplex(A.T@A+source_ridge*np.eye(n),-A.T@r,cap,budget); g=np.linalg.solve(zz,z.T@(y[~q]-S[~q]@w)); pred[q]=S[q]@w+Z[q]@g; worst=max(worst,np.sqrt(np.mean((pred[q]-y[q])**2)))
 return tuple(np.round([np.sqrt(np.mean((pred-y)**2)),worst],7))
configs=list(itertools.product((0,1),repeat=4))+[tuple(np.eye(4)[i]*v) for i in range(4) for v in (.05,.25)]+[(v,)*4 for v in (.05,.25)]
wall=[]
for w in configs:
 S=transport(w); wall.append((w,*[score(S,groups[split]) for split in groups]))
for k,split in enumerate(groups,1): print(split,'baseline',wall[0][k],'leaders',[(w[0],w[k]) for w in sorted(wall,key=lambda a:a[k])[:5]])
for split in ('sensor','shift'): print('mechanism-only sensitivity',split,[(w,score(transport(w),groups[split],1e-6,1e-6)) for w in [(0,0,0,0),(1,1,1,1)]])
print('diagnostics present:',item.get('campaign_diagnostics') is not None,'nuisance rank:',np.linalg.matrix_rank(Z),'physical sensors:',len(np.unique(ij,axis=0)),'shifts:',len(shifts))
