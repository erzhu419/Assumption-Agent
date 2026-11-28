from probe import *
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

def fit_panel(rows, v, key):
    def basis(a,b,p):
        if key=='v1': return np.c_[np.ones(len(a)),a,b,a/(1+p*a*a)]
        return np.c_[np.ones(len(a)),a,np.cos(p*a)]
    def solve(p):
        f=basis(u[rows],uz[rows],p); c=np.linalg.lstsq(f,v[rows],rcond=None)[0]; return f@c-v[rows],c
    opt=min((least_squares(lambda p:solve(p)[0].ravel(),np.full(4,i),bounds=(.05,6),max_nfev=100) for i in (.7,1.7)),key=lambda r:np.linalg.norm(r.fun)); p=opt.x; _,c=solve(p)
    return p,c,lambda a,b:basis(a,b,p)@c

pan={}
for key in ['v1','v2']:
    v=C[key]; p,c,f=fit_panel(np.arange(len(u)),v,key); err=[]
    for i in range(len(u)):
        _,_,g=fit_panel(np.delete(np.arange(len(u)),i),v,key); err.extend(abs(g(u[i:i+1],uz[i:i+1])-v[i]).ravel())
    print(key,'shape',p.round(5),'rmse',np.std(f(u,uz)-v),'LOO',np.mean(err),'coef',c.round(3),flush=True); pan[key]=(p,f(x,z))

h,j=pan['v1'][1],pan['v2'][1]; R=np.c_[y,T]; F,keys=feats(x,z)
def cross(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def quad(a): return np.array([a[:,i]*a[:,j] for i,j in itertools.combinations_with_replacement(range(a.shape[1]),2)]).T
folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=281).split(x))
def score(f,alpha):
    e=[]
    for a,b in folds:
        p=Ridge(alpha=alpha).fit(f[a]/np.maximum(.01,f[a].std(0)),R[a]); pred=p.predict(f[b]/np.maximum(.01,f[a].std(0))); d=abs(pred-R[b]);e.append([d[:,0].mean(),d[:,1:].mean(),d[:,-1].mean()])
    e=np.array(e);return np.r_[e[0],e[1:].mean(0),e[1:].max(0)]
base={'x_z_xz':np.c_[x,z,cross(x,z)],'z_xz_zz':np.c_[z,cross(x,z),quad(z)],'xz_all':F,'h_z':np.c_[h,z,cross(x,z),quad(z)],'h_x':np.c_[h,x,cross(x,z),quad(z)],'hj_z':np.c_[h,j,z,cross(x,z),quad(z)],'h_j_x':np.c_[h,j,x,cross(x,z),quad(z)],'z_xz_diag':np.c_[z,x*z,x*np.roll(z,-1,1),x*np.roll(z,1,1),z*z,z*np.roll(z,-1,1)],'raw_cal':np.c_[F,s['train']['x']]}
extras={'none':np.empty((len(x),0)),'satx':x/(1+pan['v1'][0]*x*x),'cosx':np.cos(pan['v2'][0]*x),'saturations':np.c_[x/(1+pan['v1'][0]*x*x),np.cos(pan['v2'][0]*x),z/(1+z*z),np.tanh(z)],'hprod':cross(h,x),'jprod':cross(j,x),'h':h,'j':j,'hzprod':cross(h,z),'ratprod':cross(x/(1+x*x),z),'tanhprod':cross(x,np.tanh(z))}
results=[]
for k,f in base.items():
 for l,g in extras.items():
  for alpha in (.0001,.003,.03,.15,1):
   e=score(np.c_[f,g],alpha); results.append((e,k,l,alpha))
for crit in (lambda e:e[3],lambda e:max(e[3:5]),lambda e:e[0]):
 print('SORTED');
 for e,k,l,a in sorted(results,key=lambda r:crit(r[0]))[:8]:print(k,l,a,e.round(4))
