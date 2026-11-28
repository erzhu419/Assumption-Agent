import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task')
from batch_solver import _features,_calibrate,_aff,_ls,_pairs
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; x=tr['x']; n=len(x); idx=np.arange(n); a,b=_features(s,x); z=_calibrate(s['panel_a'],x); p=s['panel_b']; m=_ls(_aff(p['u']),p['v']); bb=_aff(b)@m
print('provided keys',list(s),'train keys',list(tr),'times',s['eval_times']); print('trajectory sv',np.linalg.svd(T,compute_uv=False)); print('scalar baseline coefficients',_ls(_aff(np.c_[a,b,(a*b).sum(1)]),y)); print('panel B fit RMSE',np.std(_aff(p['u'])@m-p['v']))
p=s['panel_c']; u,v=p['u'],p['v1']; np.set_printoptions(precision=4,suppress=True)
for name,fun in {'tanh+cubic':lambda w,r:np.c_[_aff(np.tanh(r*w)),w**3],'tanh+linear+cubic':lambda w,r:np.c_[_aff(np.tanh(r*w)),w,w**3],'erf+linear+cubic':lambda w,r:np.c_[_aff(__import__('scipy').special.erf(r*w)),w,w**3],'softsqrt+linear+cubic':lambda w,r:np.c_[_aff(r*w/np.sqrt(1+(r*w)**2)),w,w**3]}.items():
 def loss(r):
  q=np.linalg.qr(fun(u,r))[0]; return np.mean(((v-q@(q.T@v))/(1-(q*q).sum(1))[:,None])**2)
 opt=minimize_scalar(loss,bounds=(.2,3),method='bounded'); d=fun(u,opt.x); coef=_ls(d,v); print(name,opt.x,np.sqrt(opt.fun),'rms',np.std(d@coef-v),'coefs',coef if name.startswith('tanh') else '')
D=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
base=_aff(np.c_[a,b]); ab=(a*b).sum(1); den=[(a*a).sum(1),(b*b).sum(1),((a-b)**2).sum(1),((a+b)**2).sum(1),ab*ab]; print('pair-difference covariance equivalence',np.max(abs(((a-b)**2).sum(1)-den[0]-den[1]+2*ab)))
scalars={'baseline':np.c_[base,ab],'panelB_sum':np.c_[base,(a*bb).sum(1)],'full':np.c_[base,D(a,b)],'pair_sums_differences':np.c_[base,ab,a*a+b*b,a*a-b*b]}; states={'baseline':np.c_[base,D(a,b)],'latent':np.c_[base,D(a,b),z],'pair_sums_differences':np.c_[base,D(a,b),a*a+b*b,a*a-b*b]}
for j,d in enumerate(den):
 for c in [.03,.1,.3,1.]:
  scalars[f'ratio{j}_{c}']=np.c_[base,ab/(1+c*d)]; states[f'ratio{j}_{c}']=np.c_[base,D(a,b)/(1+c*d[:,None])]
splits=[('required',idx[52:])]+[(f'rand{i}',np.random.default_rng(i).permutation(n)[52:]) for i in range(20)]
for j in range(4):
 order=np.argsort(x[:,j]); splits += [(f'x{j}_tails',np.r_[order[:10],order[-10:]]),(f'x{j}_low',order[:20]),(f'x{j}_high',order[-20:])]
splits += [('joint_x_ood',np.argsort(np.linalg.norm(z,axis=1))[-20:]),('joint_ab_ood',np.argsort(np.linalg.norm(D(a,b),axis=1))[-20:])]
for name,ds,Y in [('y',scalars,y),('traj',states,T)]:
 rs=[]
 for key,F in ds.items():
  rows=[]
  for sl,te in splits:
   ti=np.setdiff1d(idx,te); pred=F[te]@_ls(F[ti],Y[ti]); r=np.abs(pred-Y[te]); rows.append((r.mean(),r.mean(0).max(),sl))
  rs.append((key,rows[0][0],np.median([r[0] for r in rows]),max(rows),max(rows,key=lambda r:r[1]),float(np.std(F@_ls(F,Y)-Y))))
 print(name,'baseline:',rs[0]); print(name,'top by worst split:',sorted(rs,key=lambda r:r[3])[:6])
