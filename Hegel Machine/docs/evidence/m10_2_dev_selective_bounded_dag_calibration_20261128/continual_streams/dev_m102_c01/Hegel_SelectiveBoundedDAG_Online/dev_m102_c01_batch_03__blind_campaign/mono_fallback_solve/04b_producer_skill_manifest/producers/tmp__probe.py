import numpy as np,pickle
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(x)
def affine(u,v):
 X=np.c_[np.ones(len(u)),u]; w=np.linalg.lstsq(X,v,rcond=None)[0]; return w
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.ones(len(u),bool)
for j in range(6):
 a=affine(u[keep],v[keep]); e=np.linalg.norm(np.c_[np.ones(len(u)),u]@a-v,axis=1); keep=np.argsort(e)[:44]; print('A',j,np.round(np.sort(e)[[0,22,43,44,45,46,47]],6))
b=affine(s['panel_b']['u'],s['panel_b']['v']);print('A map',a);print('B map',b)
for p,name in [(s['panel_c'],'v1'),(s['panel_c'],'v2')]:
 u,v=p['u'],p[name]
 for family in ['tanh','sin','rat','ratodd','soft','square','tanhplus','ratplus']:
  def feat(z,h):
   t={'tanh':lambda:np.tanh(h*z),'sin':lambda:np.sin(h*z),'rat':lambda:z*z/(1+h*z*z),'ratodd':lambda:z/(1+h*z*z),'soft':lambda:z/np.sqrt(1+h*z*z),'square':lambda:z*z,'tanhplus':lambda:np.tanh(h*z),'ratplus':lambda:z*z/(1+h*z*z)}[family]()
   extra=(family in ('rat','square','tanhplus','ratplus'))
   return np.c_[np.ones(len(z)),z,t] if extra else np.c_[np.ones(len(z)),t]
  def loss(h):
   Q=np.linalg.qr(feat(u,h))[0]; return np.sqrt(np.mean(((v-Q@(Q.T@v))/(1-(Q*Q).sum(1,keepdims=True)))**2))
  opt=minimize_scalar(loss,bounds=(.02,3),method='bounded');print(name,family,round(opt.x,5),round(opt.fun,6))
coords={'x':x,'a':x@a[1:]+a[0],'ai':(x-a[0])@np.linalg.pinv(a[1:]),'b':x@b[1:]+b[0]};coords['ab']=coords['a']@b[1:]+b[0];coords['aib']=coords['ai']@b[1:]+b[0];
folds=[(np.arange(52),np.arange(52,72))];rng=np.random.default_rng(456)
for _ in range(5):
 i=rng.permutation(n);folds.append((i[:52],i[52:]))
results=[]
for key,z in coords.items():
 z=z-z.mean(0);z/=np.sqrt((z*z).sum(1).mean()); D=cdist(z,z,'sqeuclidean'); dot=z@z.T
 for width in [.05,.12,.25,.5,1,2]:
  for typ in ['rbf','m52','rq','add']:
   R=np.sqrt(5*width*D); K={'rbf':np.exp(-width*D),'m52':(1+R+R*R/3)*np.exp(-R),'rq':(1+width*D)**-1,'add':np.exp(-width*(z[:,None]-z[None,:])**2).sum(2)}[typ]+1+dot+.3*dot**2
   for ridge in [1e-6,1e-4,.003,.02]:
    E=[]
    for i,j in folds:
     e=abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+ridge*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]);E.append([e[:,0].mean(),e[:,1:].mean()]);
    E=np.array(E);results.append((E.mean(0)[0],E.max(0)[0],E[0,0],E.max(0)[1],key,width,typ,ridge))
print('ascending mean',*sorted(results)[:20],sep='\n');print('ascending gate',*sorted(results,key=lambda a:a[2])[:10],sep='\n')
