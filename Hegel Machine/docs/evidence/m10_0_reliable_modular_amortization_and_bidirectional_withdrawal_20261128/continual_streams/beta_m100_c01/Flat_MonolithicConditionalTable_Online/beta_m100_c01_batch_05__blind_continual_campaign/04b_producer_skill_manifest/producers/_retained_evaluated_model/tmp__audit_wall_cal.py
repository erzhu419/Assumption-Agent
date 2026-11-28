exec(open('/tmp/compare_walls.py').read().split("print('archived calibration sources:')")[0])
from itertools import product
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
m=M[4]
p=m._fit_calibration(ref,raw)
e=np.max(abs(m._forward(ref,p)-raw),axis=1)
print('calibration parameters\n',p)
print('calibration max-channel errors sorted',np.sort(e))
print('bank error by channel',[(i,np.max(abs(b['raw_readout'][b['bank_id']==i]-(p[i,0]+p[i,1]*b['ref_signal'][b['bank_id']==i]+p[i,5]*b['ref_signal'][b['bank_id']==i]**3)))) for i in j])
for name, design in [('affine',lambda x,i:np.c_[np.ones(len(x)),x]),('archived nonlinear',m._design)]:
 errs=[]
 for f in range(5):
  test=np.arange(len(ref))%5==f; train=~test
  if name=='archived nonlinear': pp=m._fit_calibration(ref[train],raw[train]); pred=m._forward(ref[test],pp)
  else:
   D=design(ref,0); pp=np.linalg.lstsq(D[train & (e<1)],raw[train & (e<1)],rcond=None)[0]; pred=D[test]@pp
  errs.append(np.sort(np.max(abs(pred-raw[test]),axis=1)))
 print(name,'fold errors', [np.round(u,6).tolist() for u in errs])
y=m._calibrate(s); z=s['Z_nuisance']; ridge=s['source_ridge']; rz=s['nuisance_ridge']; cap=s['u_box']; budget=s['total_budget']; nx,ny=s['nx'],s['ny']; groups=np.arange(len(y)).reshape(len(s['mu_shifts']),-1,4); rows=np.arange(len(y)); ij=s['sensor_ij']; edge=(ij[:,0]==0)|(ij[:,0]==nx-1)|(ij[:,1]==0)|(ij[:,1]==ny-1)
def transport(x,y):
 axes=[]
 for a,n,model in [('x',nx,x),('y',ny,y)]:
  args=(s['kappa_'+a],s['vel_'+a],s['L'+a]/(n-1))
  axes.append(M[2]._axis(*args) if model=='P' else M[3]._axis(*args, s['robin_'+a] if model=='R' else (0 if model=='N' else model)))
 ax,ay=axes
 L=kron(eye(ny), ax,format='csc')+kron(ay,eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); rhs=np.column_stack([v.ravel(order='F') for v in s['source_modes']]); slots=ij[:,0]+nx*ij[:,1]
 return np.vstack([splu(L+mu*I).solve(rhs)[slots] for mu in s['mu_shifts']])
def scores(S, train, test, sr=ridge):
 Z=z[train]; P=np.linalg.solve(Z.T@Z+rz*np.eye(Z.shape[1]),Z.T); A=S[train]-Z@(P@S[train]); r=y[train]-Z@(P@y[train]); H=A.T@A+sr*np.eye(S.shape[1]); w,_=m._box_simplex(H,-A.T@r,cap,budget); err=S[test]@w+z[test]@(P@(y[train]-S[train]@w))-y[test]; return np.sqrt(np.mean(err**2))
folds=[('shift',groups),('rosette',groups.transpose(1,0,2)),('edge',[rows[np.tile(edge,len(groups))]])]
print('Robin values', s['robin_x'],s['robin_y'])
for x,ywall in product('RNP',repeat=2):
 S=transport(x,ywall)
 print(x+ywall,[[round(scores(S,rows,rows,sr),7), *[round(max(scores(S,np.setdiff1d(rows,np.ravel(f)),np.ravel(f),sr) for f in ff),7) for _,ff in folds]] for sr in [ridge,1e-7]])
