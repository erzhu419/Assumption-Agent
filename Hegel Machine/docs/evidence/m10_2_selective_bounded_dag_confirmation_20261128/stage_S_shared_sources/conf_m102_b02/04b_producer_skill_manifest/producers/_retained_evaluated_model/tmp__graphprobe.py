import pickle,numpy as np,itertools as it
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
def fit(u,v):
 a=np.c_[np.ones(len(u)),u]; m=np.ones(len(u),bool)
 for _ in range(4):
  c=np.linalg.lstsq(a[m],v[m],rcond=None)[0]; e=np.linalg.norm(a@c-v,axis=1); m=e<max(.001,np.median(e)*3)
 return c
P=s['panel_c']; u=P['u']; poly=PolynomialFeatures(2); D=poly.fit_transform(u); Q=np.linalg.lstsq(D,P['v2'],rcond=None)[0]; print('C2 coefficients',list(zip(poly.get_feature_names_out(),Q)),'rmse',np.std(D@Q-P['v2']));
for q in [P['v1'],P['v2']]:
 for name,F in [('lin',D[:,:5]),('quad',D),('lin+v2',np.c_[D[:,:5],P['v2']]),('sin+lin',np.c_[D[:,:5],np.sin(2*u)]),('tanh+lin',np.c_[D[:,:5],np.tanh(2*u)]),('diagquad',np.c_[D[:,:5],u*u]),('diagcube',np.c_[D[:,:5],u*u,u**3])]:
  H=F@np.linalg.pinv(F); print('C',name,np.std(H@q-q),np.std((H@q-q)/np.maximum(1e-9,1-np.diag(H))[:,None]))
x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; A=fit(**s['panel_a']); B=fit(**s['panel_b']); za=(x-A[0])@np.linalg.inv(A[1:]); zb=za@B[1:]+B[0]; zbi=(za-B[0])@np.linalg.inv(B[1:]); C=np.linalg.lstsq(D[:,:5],P['v1'],rcond=None)[0]
sp=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(np.r_[0:20,40:72],np.arange(20,40))]
def err(p,a,b):
 e=np.abs(p-Y[b]); return np.array([e[:,0].mean(),e[:,1:].mean()])
for n,z in [('raw',x),('a',za),('ab',zb),('abi',zbi),('ac1',za@C[1:]+C[0]),('abc1',zb@C[1:]+C[0]),('aC2',poly.transform(za)@Q),('abC2',poly.transform(zb)@Q)]:
 fs=[]
 for w in [.5,1,1.5,2,3,4]:
  for nn,F in [('addtrig',np.c_[z,np.sin(w*z),np.cos(w*z)]),('polytrig',np.c_[poly.transform(z),np.sin(w*z),np.cos(w*z)]),('addtanh',np.c_[z,np.tanh(w*z)]),('polytanh',np.c_[poly.transform(z),np.tanh(w*z)]),('quadratic',poly.transform(z))]:
   K=1+F@F.T; fs.append((nn,w,K))
 for w in [.18,.35,.65,1.15,2,3.5,6,12]:
  d=cdist(z,z,'sqeuclidean'); K=np.exp(-d/(2*w*w)); fs.append(('rbf',w,K+1)); r=np.sqrt(5*d)/w; fs.append(('matern',w,1+(1+r+r*r/3)*np.exp(-r)))
 res=[]
 for nn,w,K in fs:
  for l in [1e-6,.0001,.003,.03]:
   E=np.array([err(K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+l*np.eye(len(a)),Y[a],assume_a='pos'),a,b) for a,b in sp]); res.append((E.mean(),nn,w,l,E.round(4).tolist()))
 print(n,sorted(res)[:4])
