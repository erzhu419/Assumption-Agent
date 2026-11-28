import sys,pickle,numpy as np
from scipy.linalg import cho_factor,cho_solve
from itertools import combinations_with_replacement as comb
sys.path.insert(0,'/tmp/hegel_continual_archive'); import conf_m102_a01_batch_00_solver as prior
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; z,h=prior._coordinates(s,x); n=52
D=[np.array(list(comb(range(4),k))) for k in range(1,7)]
def phi(w,k):return np.prod(w[:,D[k-1]],axis=2)
def affine(w):return np.c_[np.ones(len(w)),w]
b=s['panel_b']; B=np.linalg.lstsq(affine(b['u']),b['v'],rcond=None)[0]; c=s['panel_c']; u=c['u']
Fs=[('x',x),('z',z),('h',h),('xB',affine(x)@B),('zB',affine(z)@B),('xBi',np.linalg.solve(B[1:].T,(x-B[0]).T).T)]
fun=[('tanh',np.tanh),('sin',np.sin),('soft',lambda x:x/(1+abs(x))),('alg',lambda x:x/np.sqrt(1+x*x)),('abs',np.abs),('sqrt',lambda x:np.sqrt(1+x*x)-1),('xabs',lambda x:x*abs(x)),('sq',np.square),('cube',lambda x:x**3),('rat',lambda x:x*x/(1+x*x))]
for name,w in Fs[:]:
 for k,f,t in [('v1',lambda x:x/(1+abs(x)),.83973),('v2',lambda x:np.sqrt(1+x*x),.95913)]:
  A=lambda w:np.c_[affine(w),f(t*w)]; Fs.append((name+k,A(w)@np.linalg.lstsq(A(u),c[k],rcond=None)[0]))
_,_,V=np.linalg.svd(T[:n],full_matrices=False); Y=np.c_[y,T@V[:2].T]; I=np.eye(n)
allout=[]
def test(label,F,p):
 F=F/np.maximum(np.sqrt(np.mean(F[:n]**2,axis=0)),1e-12); G=F/np.sqrt(p); K=G@G[:n].T; C=cho_solve(cho_factor(I+K[:n]),I); a=C@Y[:n]; E=a/C.diagonal()[:,None]; P=K[n:]@a; err=np.abs(P-Y[n:]); L=np.abs(E); allout.append((label,L.mean(0),err.mean(0),P))
for name,w in Fs:
 for dec in [2,4,8,16,32]:
  for ridge in [.002,.02,.2]:
   F=np.c_[affine(w),*[phi(w,k) for k in range(2,7)]]; p=np.r_[np.full(5,1e-8),*[np.full(len(D[k-1]),ridge*dec**(k-2)) for k in range(2,7)]];test((name,'poly',dec,ridge),F,p)
 if name not in ['x','z','h','zB','xBi']:continue
 for nm,f in fun:
  for scale in [.22,.65,1.25,2.]:
   v=f(scale*w)
   for vtype,q in [('raw',v),('mix',w+.22*v),('sub',w-.22*v)]:
    for reg in [.008,.08,.8]:
     F=np.c_[affine(w),phi(w,2),q,phi(q,2),(w[:,:,None]*q[:,None,:]).reshape(len(w),-1)];test((name,nm,scale,vtype,reg),F,np.r_[np.full(5,1e-8),np.full(10,.008),np.full(F.shape[1]-15,reg)])
     if vtype!='raw':test((name,nm,scale,vtype,'q',reg),np.c_[affine(q),phi(q,2)],np.r_[np.full(5,1e-8),np.full(10,reg*.01)])
for i in range(3):
 ranked=sorted(allout,key=lambda a:a[1][i]);print('TARGET',i);print(*[(l,np.round(e,5),np.round(v,5)) for l,e,v,p in ranked[:12]],sep='\n');
 for m in [5,20,50,100]: print('ensemble',m,np.mean(abs(np.mean([a[3][:,i] for a in ranked[:m]],axis=0)-Y[n:,i])))
 print('best validation only',[(l,np.round(e[i],5),np.round(v[i],5)) for l,e,v,p in sorted(allout,key=lambda a:a[2][i])[:3]])
for name,w in Fs[:6]:
 C=np.linalg.lstsq(np.c_[affine(w),phi(w,2)][:n],Y[:n],rcond=None)[0]; print(name,'quadratic',np.round(C,3).tolist())
