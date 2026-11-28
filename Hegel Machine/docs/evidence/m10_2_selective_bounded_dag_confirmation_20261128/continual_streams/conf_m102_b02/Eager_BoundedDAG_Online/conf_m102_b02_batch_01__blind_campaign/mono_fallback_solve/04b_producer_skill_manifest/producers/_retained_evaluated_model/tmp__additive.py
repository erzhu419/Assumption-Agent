import numpy as np,pickle,itertools as it
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];d=np.load('/tmp/coords.npz');z=d['z'];b=d['erf0']; Y=np.c_[s['train']['y'],s['train']['traj']];F=[(np.arange(52),np.arange(52,72))]+list(KFold(5,shuffle=True,random_state=901).split(z));rows=[]
def ker(v,h,style):
 r=abs(v[:,None]-v)*h
 if style==0:return np.exp(-r*r/2)
 if style==1:return (1+r+r*r/3)*np.exp(-r)
 if style==2:return 1/(1+r*r)
def powers(a,h,style):return np.array([ker(v,h,style) for v in a.T])
for use in ['z','b','zb','zzb','wz','aug','bb','bz','mix']:
 if use=='z':a=z
 if use=='b':a=b
 if use=='zb':a=np.c_[z,b]
 if use=='zzb':a=np.c_[z,z,b]
 if use=='wz':a=np.c_[z,b*.5]
 if use=='aug':a=np.c_[z,b,z+b,z-b]
 if use=='bb':a=np.c_[b,b[:,0]+b[:,1],b[:,1]+b[:,3],b[:,2]-b[:,0],b[:,3]-b[:,1]]
 if use=='bz':a=np.c_[z,d['gauss0']]
 if use=='mix':a=np.c_[z,b,np.tanh(2*z),np.tanh(2*b)]
 st=a[:52].std(0);v=a/st;L=v@v.T/v.shape[1]
 for h,style in it.product([.25,.5,1,2,3,4,6],[0,1]):
  G=powers(v,h,style);k=G.mean(0);pair=k*k-(G*G).sum(0)/len(G)**2;R=np.prod(G,axis=0)**(1./len(G));
  for ix,K in enumerate([1+L+k,1+L+k+pair,1+.1*L+k+pair+k**3,1+.1*L+k+R,1+.1*L+k+.2*pair,1+.1*L+k+pair+R,(1+L)**2+2*k+2*pair]):
   for reg in [1e-6,1e-4,.001,.01]:
    e=[]
    for i,j in F:
     p=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i]);r=abs(p-Y[j]);e.append([r[:,0].mean(),r[:,1:].mean()]);
    e=np.array(e);rows.append((use,h,style,ix,reg,*e[0],*e[1:].mean(0),*e[1:].max(0)))
for ind in [(5,6),(7,8),(5,),(6,)]:
 print('RANK',ind);seen=set()
 for v in sorted(rows,key=lambda q:sum(q[k] for k in ind)):
  if v[0] in seen:continue
  seen.add(v[0]);print(v[:5],np.round(v[5:],5))
  if len(seen)==6:break
