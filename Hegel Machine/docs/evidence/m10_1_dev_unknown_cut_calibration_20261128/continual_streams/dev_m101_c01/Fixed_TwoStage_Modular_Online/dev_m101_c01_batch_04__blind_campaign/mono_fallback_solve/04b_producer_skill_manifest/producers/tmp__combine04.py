from discover04 import *
from scipy.optimize import minimize_scalar
A,B,C,D=fit_all(s); x=s['train']['x']; t=s['eval_times']; y=s['train']['y']; z=s['train']['traj']; print('times',t,'sv',np.linalg.svd(z,compute_uv=False));
for key,funcs in [('v1',{'softsign':lambda x,p:x/(1+p[0]*abs(x)), 'signedexp':lambda x,p:np.sign(x)*(-np.expm1(-p[0]*abs(x))), 'signedlog':lambda x,p:np.sign(x)*np.log1p(p[0]*abs(x)), 'xexp':lambda x,p:x*np.exp(-p[0]*abs(x))}),('v2',{'sqrt':lambda x,p:np.sqrt(1+p[0]*x*x)-1})]:
 u=s['panel_c']['u']; v=s['panel_c'][key]
 for name,func in funcs.items():
  f,p=projected(u[:18],v[:18],func,[1],([.01],[8])); print('altC',key,name,p,np.sqrt(np.mean((f(u[18:])-v[18:])**2)))
random=np.random.default_rng(91); cv=[(np.arange(56),np.arange(56,72))]+[(i[:56],i[56:]) for i in (random.permutation(72) for _ in range(7))]; T=np.c_[y,z]
def prod(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def nl(u):
 return np.c_[u/(1+abs(u)),np.sqrt(1+u*u)-1]
def run(H,i,j,p):
 return H[j]@ls(np.r_[H[i],p*np.eye(H.shape[1])],np.r_[T[i],np.zeros((H.shape[1],T.shape[1]))])
rows=[]
for pan in ('noB','B'):
 for mode in ('x','a','i'):
  u={'x':x,'a':A[0]+x*A[1],'i':(x-A[0])/A[1]}[mode]; u=aff(u)@B if pan=='B' else u
  for kind,F in [('C',C(u)),('D',D(u)),('CD',np.c_[C(u),D(u)]),('nl',nl(u))]:
   for model in ('mix','mixQ','newQ','sep','sepQ','sepI','cubic'):
    H=poly(u,2); J=F if kind!='nl' else F; G=np.c_[J,prod(u,J)]; E=poly(J,2)[:,1+J.shape[1]:]; K=np.c_[u*J[:,:4],u*J[:,-4:]]
    if model=='mix': G=G
    if model=='mixQ': G=np.c_[G,E]
    if model=='newQ': G=np.c_[J,E]
    if model=='sep': G=np.c_[J,K]
    if model=='sepQ': G=np.c_[J,K,J*J]
    if model=='sepI': G=np.c_[J,np.array([u[:,a]*J[:,b] for b in range(J.shape[1]) for a in range(4) if a!=b%4]).T]
    if model=='cubic': G=np.c_[G,poly(u,3)[:,15:]]
    for p in [.015,.05,.12,.24]:
     HH=np.c_[H,G]; es=[]
     for i,j in cv:
      P=run(HH,i,j,p); e=abs(P-T[j]); es.append([e[:,0].mean(),e[:,1:].mean()])
     rows.append(((pan,mode,kind,model,p),np.array(es)))
for j in range(2):
 print('top mean',j)
 for name,e in sorted(rows,key=lambda r:r[1][:,j].mean())[:11]: print(name,np.round(e[:,j],5),round(e[:,j].mean(),5))
 print('top worst',j)
 for name,e in sorted(rows,key=lambda r:r[1][:,j].max())[:6]: print(name,np.round(e[:,j],5),round(e[:,j].mean(),5))
