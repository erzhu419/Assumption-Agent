import numpy as np,pickle,runpy
from scipy.optimize import least_squares
from itertools import combinations
A=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py'); align,branch,ones,ls=(A[k] for k in ('_align','_branch','_ones','_ls'))
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=(s['train'][k] for k in ('x','y','traj')); a,b=align(s['panel_a']); z=(x-a)/b;c,d=(branch(s['panel_c'],z,k) for k in (1,2)); dot=(c*d).sum(1); cd=(c[:,:,None]*d[:,None,:]).reshape(len(x),-1)
np.set_printoptions(precision=4,suppress=True)
print('align',a,b)
for k in (1,2):
 u=s['panel_c']['u'];r=branch(s['panel_c'],u,k)-s['panel_c']['v'+str(k)];print('branch',k,np.sqrt(np.mean(r*r,axis=0)))
u,v=(s['panel_b'][k] for k in ('u','v'));print('b affine matrix and residual',ls(ones(u),v),np.std(ones(u)@ls(ones(u),v)-v,axis=0));print('b corr',np.corrcoef(u.T,v.T)[:4,4:]);
base=np.c_[d,c]; baseline=ones(np.c_[base,dot]);print('baseline scalar coefficients',ls(baseline,y));
for nam,D in {'z':z,'cd base':base,'archived':np.c_[base,dot],'full cross':np.c_[base,cd],'matching cross':np.c_[base,c*d],'squares':np.c_[base,dot,c*c,d*d,z*z],'z quadratic':np.c_[z,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]],'branch paired squares':np.c_[base,*[w[:,i]*w[:,j] for w in (c,d) for i in range(4) for j in range(i,4)],cd]}.items():
 D=ones(D); e=D[52:]@ls(D[:52],np.c_[y,t][:52])-np.c_[y,t][52:];f=D@ls(D,np.c_[y,t])-np.c_[y,t]; print(nam,'val',np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])), 'train',np.mean(abs(f[:,0])),np.mean(abs(f[:,1:])))
res=y-baseline@ls(baseline,y);print('largest residuals indices, y, r, z, dot'); ii=np.argsort(abs(res))[-12:];print(np.c_[ii,y[ii],res[ii],z[ii],dot[ii]])
