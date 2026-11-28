from ops14 import *
import glob,importlib.util
from itertools import combinations_with_replacement
f=glob.glob('/tmp/hegel_continual_archive/*_09_solver.py')[0]; sp=importlib.util.spec_from_file_location('arch',f); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
tr=s['train'];x=np.r_[tr['x'],s['test']['x']]; z=m._latent(s['panel_a'],x); b=s['panel_b']; c=s['panel_c']; B=op(b['u'],b['v'],'cos',False)[3]; C=op(c['u'],c['v1'],'tanh_sin',False)[3]; h=C(B(z)); Y=np.c_[tr['y'],tr['traj']]; n=72
v=np.c_[z,h]; names=[f'{a}{i}' for a in 'zh' for i in range(4)]; ix=list(combinations_with_replacement(range(8),2)); Q=D(v,np.column_stack([v[:,i]*v[:,j] for i,j in ix])); names=['1',*names,*[names[i]+'*'+names[j] for i,j in ix]]
W=ls(Q[:52],Y[:52]); print('condition',np.linalg.cond(Q[:52]),'fit',np.mean(abs(Q[:52]@W-Y[:52]),0),'hold',np.mean(abs(Q[52:n]@W-Y[52:]),0));print('temporal sv',np.linalg.svd(Y[:,1:],compute_uv=False));print('coefficients:');
for nm,w,p in zip(names,W,np.std(Q[:52],0)):print(nm,np.round(w[[0,1,3,5,8]],4),'impact',np.round(w[[0,8]]*p,4))
L=D(v,np.sum(z*h,1)); W0=ls(L[:52],Y[:52]);print('linear plus trace y/end',W0[:,[0,-1]]); print('first trajectory rows',np.round(tr['traj'][:5],4));
