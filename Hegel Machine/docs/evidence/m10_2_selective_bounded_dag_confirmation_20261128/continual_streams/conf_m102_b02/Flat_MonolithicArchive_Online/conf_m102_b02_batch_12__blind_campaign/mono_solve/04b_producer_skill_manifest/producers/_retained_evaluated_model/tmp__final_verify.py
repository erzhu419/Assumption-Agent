import sys
sys.path.insert(0,'/tmp')
from stress import m,s,t,y,xx,slices,split,np
for name,i,j in slices:
 p=m.solve_batch(split(i,j)); ph=np.c_[p['y_hat'],p['state_pred']]; e=np.abs(ph-y[j]); vals=np.r_[e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()]; print(name,vals); assert max(vals)<.012
 end=ph[:,-1]; assert np.allclose(p['diagnostics_vec'],[ph[:,0].mean(),ph[:,0].std(),end.mean(),end.std()])
z,h=m._coords(s,xx); i,j=slices[0][1:]; base=np.c_[np.ones(len(z)),z]; E=np.abs(base[j]@m._ls(base[i],y[i])-y[j]); print('calibration-only 52/20 y/state MAE',E[:,0].mean(),E[:,1:].mean())
