import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z_tr'];y=s['train']['y'];a,b,c,d=z.T
L=np.c_[np.ones(len(z)),z];I=np.c_[a*c,b*d];pairs=np.c_[a*a,a*b,a*c,a*d,b*b,b*c,b*d,c*c,c*d,d*d]; tri=np.c_[a*b*c,a*b*d,a*c*d,b*c*d]
links={'tanh':np.tanh,'atan':np.arctan,'erf':__import__('scipy.special',fromlist=['erf']).erf,'softsign':lambda x:x/(1+abs(x)),'root':lambda x:x/np.sqrt(1+x*x),'sin':np.sin,'rational-odd':lambda x:x/(1+x*x),'asinh':np.arcsinh,'cubic-saturation':lambda x:x/np.cbrt(1+abs(x)**3),'exp-saturation':lambda x:np.sign(x)*(-np.expm1(-abs(x)))}
splits=[(np.r_[0:52],np.r_[52:72])];rng=np.random.default_rng(24)
for _ in range(4):
 ix=rng.permutation(72);splits.append((ix[20:],ix[:20]))
for name,f in links.items():
 for mode in ('independent','linked','interaction-in-index','quadratic','triple'):
  scores=[]
  for tr,te in splits:
   q0=np.linalg.lstsq(np.c_[L,I][tr],y[tr],rcond=None)[0]
   def predict(p):
    q=L@p[:5]
    if mode=='interaction-in-index':q=q+I@p[6:8];return q+p[5]*f(p[8]*q)
    return q+p[5]*f(p[6]*q)+I@p[7:9]
   if mode in ('linked','interaction-in-index'):
    starts=[np.r_[q0[:5]*k,m,1.7,q0[5:]] if mode=='linked' else np.r_[q0[:5]*k,m,q0[5:],1.7] for k,m in [(1,.3),(1,-.3),(.7,.6)]]
    opts=[least_squares(lambda p:(predict(p)-y)[tr],p,max_nfev=120) for p in starts];opt=min(opts,key=lambda x:x.cost);pred=predict(opt.x); pars=opt.x
   else:
    base=np.c_[L,I] if mode=='independent' else np.c_[L,pairs] if mode=='quadratic' else np.c_[L,I,tri]
    def solve(p):
     A=np.c_[base,f(L@p)]; w=np.linalg.lstsq(A[tr],y[tr],rcond=None)[0];return A@w,w
    opts=[least_squares(lambda p:(solve(p)[0]-y)[tr],q0[:5]*k,bounds=(-6,6),max_nfev=100) for k in (.6,1.4,2.)];opt=min(opts,key=lambda x:x.cost);pred,w=solve(opt.x);pars=np.r_[opt.x,w]
   e=pred-y;scores.append([np.sqrt(np.mean(e[tr]**2)),np.mean(abs(e[te])),np.max(abs(e[te]))]);
   if len(scores)==1: pfirst=pars
  scores=np.array(scores);print(name,mode,'train/public/worst/row',np.round([scores[0,0],scores[0,1],max(scores[:,1]),max(scores[:,2])],5),'params',np.round(pfirst,3),flush=True)
