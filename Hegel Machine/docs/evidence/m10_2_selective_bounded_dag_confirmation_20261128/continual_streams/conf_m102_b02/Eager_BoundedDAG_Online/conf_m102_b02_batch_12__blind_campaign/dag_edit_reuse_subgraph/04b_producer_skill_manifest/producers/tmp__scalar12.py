import sys,pickle,numpy as np
from scipy.special import erf
from scipy.linalg import solve
from scipy.spatial.distance import cdist
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_b02_batch_05_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=len(s['train']['x']); z,b=old.parts(s); z=z[:n]; b=b[:n]; y=s['train']['y']; rng=np.random.default_rng(700)
ids=rng.permutation(n); folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(ids,j),j) for j in np.array_split(ids,4)]
print('y',y.round(4),'range z,b',z.min(0),z.max(0),b.min(0),b.max(0))

def sc(k,r):
 return np.array([np.mean(abs(y[j]-k[np.ix_(j,i)]@solve(k[np.ix_(i,i)]+r*np.eye(len(i)),y[i],assume_a='pos'))) for i,j in folds])
def pair(x,w):return (x[:,:,None]*w[:,None,:]).reshape(n,-1)

def phi(x):return {'tanh':np.tanh(x),'erf':erf(x),'atan':np.arctan(x),'soft':x/np.sqrt(1+x*x),'rat':x/(1+x*x),'sin':np.sin(x),'asinh':np.arcsinh(x),'raw':x}
res=[]
for scale in [1.,1.25,1.6,2.15,2.5,3.,3.5]:
 for zk in [0,.15,.3,.5,1.]:
  for cname,c in [('sep',b),('plus',b+zk*z),('minus',b-zk*z),('sym',np.c_[b+zk*z,b-zk*z])]:
   for name,h in phi(scale*c).items():
    l=np.c_[z,h]; zz=z@z.T/4; hh=h@h.T/h.shape[1]; zh=zz*hh; ds=cdist(l,l,'sqeuclidean')/l.shape[1]; diagz=(z*z)@(z*z).T/4; diagh=(h*h)@(h*h).T/h.shape[1];
    for mode,q in enumerate([0,hh**2,zh,diagz+diagh,zh+diagh,zh+diagz+diagh,zh+hh**2,hh**2+zz**2+2*zh,.15*np.exp(-ds),.15*np.exp(-ds)+zh+hh**2]):
     k=1+zz+hh+q
     for ridge in [1e-5,3e-4,.003,.02]:
      v=sc(k,ridge);res.append((v.mean()+.4*v.max(),scale,zk,cname,name,mode,ridge,v,k))
res.sort(key=lambda t:t[0]); print('BEST');print(*[r[:-1] for r in res[:20]],sep='\n');print('BEST PUBLIC');print(*[r[:-1] for r in sorted(res,key=lambda t:t[-2][0])[:12]],sep='\n');print('ENSEMBLES');
for count in (3,10,30,100):
 k=sum(r[-1] for r in res[:count])/count; print(count,sc(k,.001))
for label,h in [('z',z),('b',b),('tanh',np.tanh(2.15*b)),('erf',erf(2.15*b)),('atan',np.arctan(2.15*b)),('soft',2.15*b/np.sqrt(1+(2.15*b)**2))]:
 f=np.c_[np.ones(n),z,h]; w=np.linalg.lstsq(f,y,rcond=None)[0]; print(label,w.round(4),np.mean(abs(f@w-y)),sc(f@f.T,.001))
