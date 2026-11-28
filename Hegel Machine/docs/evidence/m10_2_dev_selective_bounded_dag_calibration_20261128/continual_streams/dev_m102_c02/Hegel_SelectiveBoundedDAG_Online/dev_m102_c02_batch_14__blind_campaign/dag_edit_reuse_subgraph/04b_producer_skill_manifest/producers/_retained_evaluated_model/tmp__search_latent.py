exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from itertools import combinations,product
from scipy.linalg import qr
m=load('/holdout_task/node_00.py'); z=m._latent(a,x); y=s['train']['y']; T=s['train']['traj']; n=52
B=m._panel_b(b,z); norm=lambda f: f/np.maximum(f[:n].std(0),1e-9)
U,S,V=np.linalg.svd(T[:n],full_matrices=False); A=T@V[:3].T
C=[]
for key in ['v1','v2']:
 p=s['panel_c'];C.append(new._op(p,key,z,'c') if key=='v1' else np.c_[np.ones(len(z)),z,z*z,z**3]@ls(np.c_[np.ones(len(p['u'])),p['u'],p['u']**2,p['u']**3],p[key]))
F={'z':z,'x':x,'b':B,'p':C[0],'q':C[1],'s':np.sin(2.15*z),'t':np.tanh(1.25*z),'r':z/(1+1.25*z*z),'h':z*z/(1+1.25*z*z),'g':z**3/(1+1.25*z*z),'z2':z*z,'z3':z**3,'z4':z**4,'z5':z**5}; D={}
for k,f in F.items():
 for j in range(4): D[k+str(j)]=f[:,j]
for k,l in combinations(['z','b','p','q','s','t','r','h','g','z2','z3'],2):
 for j,i in product(range(4),repeat=2):D[f'{k}{j}*{l}{i}']=F[k][:,j]*F[l][:,i]
for k in ['z','b','p','q','s','r','g']:
 for j,i in combinations(range(4),2):D[f'{k}{j}*{k}{i}']=F[k][:,j]*F[k][:,i]
for a1,b1,c1 in combinations(range(4),3):D[f'z{a1}*z{b1}*z{c1}']=z[:,a1]*z[:,b1]*z[:,c1]
for j,i in combinations(range(4),2):
 for k in [-1,1]:
  w=z[:,j]+k*z[:,i]
  for tag,f in [('sin',np.sin(2.15*w)),('rat',w/(1+1.25*w*w)),('mix',w**2/(1+1.25*w*w)),('cub',w**3/(1+1.25*w*w))]:D[f'{tag}(z{j}{k:+}z{i})']=f
names=list(D); d=np.column_stack(list(D.values())); d=(d-d[:n].mean(0));d=norm(d)
for base in [['z','b'],['z','b','p','q'],['z','z2','z3','z4','z5','r','h'],['z','b','p','q','s','r','h','g']]:
 basef=np.column_stack([F[k] for k in base]); print('BASE',base)
 for ia,Y in enumerate([A[:,0],y,y-A[:,0]*np.dot(y[:n],A[:n,0])/np.dot(A[:n,0],A[:n,0])]):
  src=lin(basef); inds=[]; print('output',ia)
  for step in range(12):
   df=np.c_[src,d[:,inds]]; co=ls(df[:n],Y[:n]); res=Y[:n]-df[:n]@co; ort=d[:n]-df[:n]@ls(df[:n],d[:n]); os=np.linalg.norm(ort,axis=0); merit=np.abs(ort.T@res)/np.maximum(os,1e-7);merit[os<1e-6]=0; ix=np.argmax(merit); print(step,round(np.mean(np.abs(df[n:]@co-Y[n:])),4),round(np.linalg.norm(res)/np.sqrt(n),4),names[ix],end='; '); inds.append(ix)
  print()
for base in [['z'],['z','b'],['z','b','p','q'],['z','z2','b','p','q']]:
 Z=norm(np.column_stack([F[k] for k in base])); Y=np.c_[y,A/np.maximum(A[:n].std(0),0.05)];
 from scipy.spatial.distance import cdist
 from scipy.linalg import solve
 for kind in ['rbf','matern','poly']:
  L=[];ds=cdist(Z,Z[:n],'sqeuclidean')/Z.shape[1]
  for ell,reg in product([.4,.7,1,1.6,2.5,4,7],[1e-6,1e-4,1e-3,.01,.05]):
   H=np.exp(-ds/ell**2) if kind=='rbf' else ((1+np.sqrt(ds)*ell)*np.exp(-np.sqrt(ds)*ell) if kind=='matern' else (1+Z@Z[:n].T/Z.shape[1])**int(ell+2)); K=H[:n]+reg*np.eye(n); inv=solve(K,np.eye(n),assume_a='pos'); w=inv@(Y[:n]-Y[:n].mean(0)); loo=w/np.diag(inv)[:,None]; E=Y[n:]-(H[n:]@w+Y[:n].mean(0));L.append((np.mean(abs(loo),axis=0)[:2],ell,reg,np.mean(abs(E),axis=0)[:2]))
  print('kernel',base,kind,sorted(L,key=lambda q:sum(q[0]))[:2])
