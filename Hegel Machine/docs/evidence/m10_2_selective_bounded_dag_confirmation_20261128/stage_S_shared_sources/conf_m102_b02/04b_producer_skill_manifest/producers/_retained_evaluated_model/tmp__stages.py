import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def col(z):return np.c_[np.ones(len(z)),z]
def fit(u,v):
 a=col(u); m=np.ones(len(u),bool)
 for _ in range(4):
  c=ls(a[m],v[m]); e=np.linalg.norm(a@c-v,axis=1);m=e<max(.001,np.median(e)*3)
 return c
A=fit(**s['panel_a']); B=fit(**s['panel_b']); P=s['panel_c']; u=P['u']; v=P['v1']; w=P['v2']
def vp(fun,p,y, verbose=True):
 def err(p):
  d=fun(p,u); return (d@ls(d,y)-y).ravel()
 q=least_squares(err,p,max_nfev=250,gtol=1e-12); d=fun(q.x,u); c=ls(d,y)
 if verbose: print(fun.__name__,q.x,np.std(q.fun),np.std(q.fun.reshape(y.shape)/(1-np.diag(d@np.linalg.pinv(d)))[:,None]));print(c)
 return lambda z:fun(q.x,z)@c

def tan(p,z):return col(np.tanh(p*z))
def tanoff(p,z):return col(np.tanh(p[:4]*z+p[4:]))
def lintan(p,z):return col(np.c_[z,np.tanh(p*z)])
def rat(p,z):return col((z+p[0]*z*z)/(1+p[1]*z*z))
def rat2(p,z):return col((z+p[0]*z*z)/(1+p[1]*z*z+p[2]*z**4))
def quadcube(p,z):return col(z+p[0]*z*z+p[1]*z**3+p[2]*z**4)
def qtanh(p,z):return col(np.tanh(p[1]*(z+p[0]*z*z)))
for f,p in [(tan,[2]),(tan,[2]*4),(tanoff,[2]*4+[0]*4),(lintan,[2]),(lintan,[2]*4)]:
 c=vp(f,p,v)
 if f==tan and len(p)==1:C=c
for f,p in [(rat,[.44,0]),(rat2,[.44,0,0]),(quadcube,[.44,0,0]),(qtanh,[.44,.3])]:
 c=vp(f,p,w)
 if f==rat:D=c
x=s['train']['x']; T=s['train']['traj']; y=s['train']['y']; time=s['eval_times']; t=np.c_[-np.expm1(-1.8*time),-np.expm1(-.4*time)]; Y=np.c_[y,T]; amp=T@np.linalg.pinv(t.T); print('amp correlation',np.corrcoef(np.c_[y,amp].T));
a=(x-A[0])@np.linalg.inv(A[1:]); af=col(x)@A; b=col(a)@B; bf=col(af)@B; br=col(x)@B
sp=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(np.r_[0:20,40:72],np.arange(20,40))]
coords={}
for n,z in [('x',x),('a',a),('af',af),('b',b),('bf',bf),('br',br)]:
 for k,d in [('',z),('C',C(z)),('D',D(z)),('CD',np.c_[C(z),D(z)]),('zCD',np.c_[z,C(z),D(z)]),('CB',col(C(z))@B),('DB',col(D(z))@B)]:coords[n+k]=d

def errors(F):
 E=[]
 for i,j in sp:
  pred=F[j]@ls(F[i],Y[i]); e=np.abs(pred-Y[j]); E.append([e[:,0].mean(),e[:,1:].mean()])
 E=np.array(E); return E.mean(),E.round(5).tolist()
R=[]
for n,z in coords.items():
 F=col(z); R.append((*errors(F),n,'lin'))
 if z.shape[1]>4:continue
 pol=PolynomialFeatures(2).fit_transform(z); R.append((*errors(pol),n,'quad'))
 for r in [.7,1.2,1.8,2.4,3.2]:
  for k,f in [('tanh',np.tanh(r*z)),('sin',np.sin(r*z)),('sin2',np.c_[np.sin(r*z),np.cos(r*z)]),('rat',z/(1+r*z*z)),('cubic',z**3)]:
   for name,d in [('lin',F),('quad',pol)]:R.append((*errors(np.c_[d,f]),n,name+k+str(r)))
print('best joint',*sorted(R)[:15],sep='\n'); print('best by output',*[sorted(R,key=lambda r:np.mean(np.array(r[1])[:,j]))[:5] for j in (0,1)],sep='\n')
