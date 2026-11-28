exec(open('/tmp/readout.py').read().replace("(cal(zz)*c).sum(2)","xs").split('from sklearn')[0].replace('return z\nz=inv','return z\nc=np.stack([fit(np.c_[ones(u),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3],v[:,j]) for j in range(4)]); oldinv=inv\ndef inv(x,c):\n z=(x-c[:,0])/c[:,1]\n for _ in range(60):\n  w=np.roll(z,-1,axis=1); z-=(c[:,0]+c[:,1]*z+c[:,2]*w+c[:,3]*z*w+c[:,4]*z**3-x)/(c[:,1]+c[:,3]*w+3*c[:,4]*z*z)\n return z\nz=inv'))
print('B',b,'y-v-traj',np.std(Y[:,0]-np.c_[ones(x),T]@fit(np.c_[ones(x),T],Y[:,0])))
p=s['panel_c']; C2=fit(np.c_[ones(p['u']),np.cos(2.4*p['u'])],p['v2']);print('C2 coefficients',C2)
def cross(u):return np.column_stack([u[:,i]*u[:,j] for i,j in it.combinations(range(4),2)])
def fset(z,k):
 z1=z; nxt=np.roll(z,-1,axis=1); W=quad(z)@q; C=np.c_[ones(z),np.cos(2.4*z)]@C2; B=z@b[1:]+b[0];
 return {'basic':np.c_[quad(z),z**3,z**4], 'phys':np.c_[ones(z),z,B,W,C,np.tanh(z),cross(np.tanh(z)),np.sin(k*z),np.cos(k*z),cross(np.sin(k*z))], 'fourier':np.c_[ones(z),np.sin(k*z),np.cos(k*z),*[f(k*(z[:,i]+sg*z[:,j])) for i,j in it.combinations(range(4),2) for sg in [-1,1] for f in [np.sin,np.cos]]], 'hybrid':np.c_[quad(z),np.sin(k*z),np.cos(k*z),np.sin(k*cross(z)),np.tanh(k*cross(z)),cross(np.tanh(k*z)),C,W]} 
from sklearn.model_selection import KFold
from sklearn.linear_model import OrthogonalMatchingPursuit
splits=[(np.arange(52),np.arange(52,n)),*KFold(4,shuffle=True,random_state=4).split(x)]; rates=np.array([.4153,1.8082]); ts=-np.expm1(-t[:,None]*rates); R=np.c_[Y[:,0],fit(ts,T.T).T]; out=[]
D.update({'Aexact':z,'raw_forward_A':np.sum(np.stack([np.ones_like(x),x,np.roll(x,-1,axis=1),x*np.roll(x,-1,axis=1),x**3],axis=2)*c,axis=2)})
for nm,u in D.items():
 if nm.endswith('C'): continue
 for k in [.5,.8,1.,1.2,1.6,2.,2.4,2.8,3.2,4.,5.,6.]:
  for kind,f in fset(u,k).items():
   if k!=.5 and kind=='basic':continue
   for reg in [.0001,.01,.2,2, ('omp',5), ('omp',10), ('omp',16)]:
    es=[]
    for I,J in splits:
     mu=f[I].mean(0); std=f[I].std(0); mask=std>1e-8; L=(f[I][:,mask]-mu[mask])/std[mask]; V=(f[J][:,mask]-mu[mask])/std[mask]; m=R[I].mean(0); h=R[I]-m
     if isinstance(reg,tuple):
      import warnings
      with warnings.catch_warnings():
       warnings.simplefilter('ignore'); P=OrthogonalMatchingPursuit(n_nonzero_coefs=min(reg[1],L.shape[1]),fit_intercept=False).fit(L,h).predict(V)+m
     else:P=V@L.T@np.linalg.solve(L@L.T+reg*np.eye(len(I)),h)+m
     es.append([np.mean(abs(P[:,0]-R[J,0])),np.mean(abs(P[:,1:]@ts.T-T[J]))])
    out.append((np.mean(es,0),np.max(es,0),nm,k,kind,reg))
for a in range(2):print('ENDPOINT',a,*sorted(out,key=lambda d:d[0][a])[:16],sep='\n')
