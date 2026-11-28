import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_b03_batch_00_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; one=old._ones; ls=old._ls
np.set_printoptions(precision=5,suppress=True)
p=s['panel_a']; a,b=old._align(p); r=p['v']-(a+b*p['u']); good=np.linalg.norm(r,axis=1)<.1; print('diag',a,b,'inlier',sum(good),np.std(r[good],axis=0)); print('full inlier',ls(one(p['u'][good]),p['v'][good]));
p=s['panel_b']; B=ls(one(p['u']),p['v']); print('B gram',B[1:]@B[1:].T)
p=s['panel_c']; U=p['u'];
for j in (1,2):
 v=p['v'+str(j)]; print('archived panel C',j,np.sqrt(np.mean((v-old._branch(p,U,j))**2)))
 for name,f,start in [('soft',lambda w:w/(1+abs(w)),.84),('tanh',np.tanh,1),('asinh',np.arcsinh,1),('sin',np.sin,1.2),('cos',np.cos,1.27),('rational',lambda w:1/(1+w*w),.7),('square',lambda w:w*w,1)]:
  def d(z,q):return one(np.c_[z,f(q[0]*z+q[1:])])
  def r(q):
   D=d(U,q);return (D@ls(D,v)-v).ravel()
  opt=least_squares(r,[start,0,0,0,0], max_nfev=180); print(j,name,np.std(opt.fun),opt.x)
 def show(z):return old._branch(p,z,j)
 print('std panelC',v.std(0))
H=lambda p:-np.expm1(-t[:,None]*p)
opt=least_squares(lambda p:(H(p)@ls(H(p),T[:52].T)-T[:52].T).ravel(),[2,.4],bounds=([.5,.03],[5,1.2])); print('rates/rmse',opt.x,np.std(opt.fun)); A=ls(H(opt.x),T.T).T; print('latent std',A.std(0));
def score(D,targets=np.c_[y,A], ridge=0):
 tr=D[:52]; coef=ls(tr,targets[:52]) if ridge==0 else np.linalg.solve(tr.T@tr+ridge*np.eye(tr.shape[1]),tr.T@targets[:52]); e=D@coef-targets; return np.r_[abs(e[:52]).mean(0),abs(e[52:]).mean(0),np.mean(abs(e[52:,1:]@H(opt.x).T))]
def feat(c,d):
 cross=(c[:,:,None]*d[:,None,:]).reshape(len(c),-1); dot=np.sum(c*d,axis=1); n1=np.sum(c*c,axis=1);n2=np.sum(d*d,axis=1);return cross,dot,n1,n2
z=(x-a)/b; Bb=lambda z:one(z)@B
for nm,w in [('raw',x),('z',z),('B(z)',Bb(z)),('Binv(z)',(z-B[0])@np.linalg.inv(B[1:])),('B(x)',Bb(x))]:
 c,d=(old._branch(p,w,j) for j in (1,2)); cross,dot,n1,n2=feat(c,d)
 for desc,D in [('linear',one(np.c_[c,d,dot])),('cross',one(np.c_[c,d,cross])),('plus norms',one(np.c_[c,d,cross,n1,n2])),('include inputs',one(np.c_[w,c,d,cross,n1,n2]))]: print(nm,desc,score(D))
