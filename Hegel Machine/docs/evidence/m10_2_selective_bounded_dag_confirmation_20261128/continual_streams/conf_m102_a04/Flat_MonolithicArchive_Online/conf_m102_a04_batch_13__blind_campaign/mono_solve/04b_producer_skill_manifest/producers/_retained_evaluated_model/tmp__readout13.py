import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
from conf_m102_a04_batch_00_solver import _align,lin,one,_time_basis
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; p=s['panel_c']; u=p['u']; a,b=_align(s['panel_a']); z=(tr['x']-a)/b; i,j=np.triu_indices(4,1)
np.set_printoptions(precision=5,suppress=True)
def profile(fn,v,q,bounds):
 def r(q):
  f=fn(u,q);return (f@lin(f,v)-v).ravel()
 q=least_squares(r,q,bounds=bounds,gtol=1e-12).x;w=lin(fn(u,q),v);print('panel',q,np.std(r(q)),w,sep='\n');return fn(z,q)@w
g=profile(lambda u,q:one(u/(1+q[0]*abs(u))+q[1]*u),p['v1'],[.83,-.16],([.3,-.4],[2,.4]))
h=profile(lambda u,q:one(np.c_[np.tanh(q[0]*u),u[:,i]*u[:,j]]),p['v2'],[1.05],(.4,2.))
p=s['panel_b']; wb=lin(one(p['u']),p['v']); B=one(z)@wb; print('B residual',np.std(one(p['u'])@wb-p['v']),wb)
cross=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
sq=lambda a:np.c_[a*a,a[:,i]*a[:,j]]
F={'gh':np.c_[g,h],'rawgh':np.c_[z,g,h],'ghdot':np.c_[g,h,(g*h).sum(1)],'ghdiag':np.c_[g,h,g*h],'ghall':np.c_[g,h,cross(g,h)],'ghsq':np.c_[g,h,sq(g),sq(h)],'ghpoly':np.c_[g,h,cross(g,h),sq(g),sq(h)],'rawghall':np.c_[z,g,h,cross(g,h)],'rawghpoly':np.c_[z,g,h,cross(g,h),sq(g),sq(h)],'rawgh+rawsq':np.c_[z,g,h,sq(z)],'rawgh+rawcross':np.c_[z,g,h,cross(z,g),cross(z,h)],'rawghB':np.c_[z,g,h,B],'ghBdot':np.c_[g,h,B,(g*h).sum(1),(g*B).sum(1),(h*B).sum(1)]}
rng=np.random.default_rng(7); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(8)]; t=s['eval_times']; out=[]
for nm,f in F.items():
 for reg in (1e-7,.0001,.003,.03,.3):
  M=[]
  for perm in splits:
   a,b=perm[:52],perm[52:]; A=(f-f[a].mean(0))/np.maximum(f[a].std(0),1e-6); K=1+A@A.T; fb=_time_basis(t,tr['traj'][a]); Y=np.c_[tr['y'][a],lin(fb,tr['traj'][a].T).T]; pred=K[np.ix_(b,a)]@np.linalg.solve(K[np.ix_(a,a)]+reg*np.eye(len(a)),Y); e=abs(pred[:,1:]@fb.T-tr['traj'][b]); M.append([np.mean(abs(pred[:,0]-tr['y'][b])),e.mean(),e.mean(0).max()]);
  M=np.array(M);out.append((nm,reg,*M[0],*M.max(0),*M.mean(0)))
for k in (2,3,8,9):
 print('rank',k);print(*sorted(out,key=lambda x:x[k])[:13],sep='\n')
