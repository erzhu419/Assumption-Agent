import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def one(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
a=s['panel_a']; u=a['u']; v=a['v']; m=np.all(abs(v-u)<2,axis=1); ca=np.array([ls(one(u[m,j,None]),v[m,j]) for j in range(4)]).T
b=s['panel_b']; cb=ls(one(b['u']),b['v']); u=s['panel_c']['u']; V=[s['panel_c'][key] for key in ['v1','v2']]
funcs={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'square':np.square,'softsign':lambda z:z/(1+abs(z)),'exp':lambda z:np.exp(np.clip(z,-10,10)),'rational':lambda z:z/(1+z*z),'abs':abs}
models=[]
for k,v in enumerate(V):
 cand=[]
 for nm,g in funcs.items():
  for kind in range(3):
   def F(u,p):
    pp=np.exp(p[:4]) if kind==2 else np.exp(p[0]); sh=p[-4:] if kind else 0
    return one(g(u*pp+sh)) if kind!=1 else one(np.c_[u,g(u*pp+sh)])
   n=[1,5,8][kind]; lo=np.r_[np.full(4 if kind==2 else 1,-2.3),np.full(4 if kind else 0,-3)]; hi=np.r_[np.full(4 if kind==2 else 1,2.2),np.full(4 if kind else 0,3)]
   for freq in [0.7,1.6,3.2,5.5]:
    p0=np.r_[np.full(4 if kind==2 else 1,np.log(freq)),np.zeros(4 if kind else 0)]
    def score(p):
     D=F(u,p); return (D@ls(D,v)-v).ravel()
    f=least_squares(score,p0,bounds=(lo,hi),max_nfev=160)
    cand.append((np.std(f.fun),nm,kind,f.x,ls(F(u,f.x),v)))
 for best in sorted(cand,key=lambda t:t[0])[:7]:print(k,best)
 models.append(min(cand,key=lambda t:t[0]))
np.save('/tmp/panels.npy',np.array([ca,cb,models],dtype=object),allow_pickle=True)
