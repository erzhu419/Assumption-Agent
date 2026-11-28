from explore import *
from scipy.linalg import solve
from scipy.spatial.distance import cdist
exec(open('/tmp/operators.py').read().split("if __name__=='__main__':")[0].replace('print(', 'quiet(').replace('from explore import *','def quiet(*a):pass'))

def fitpred(K,H,T,tr,te,alpha):
 G=K[np.ix_(tr,tr)]+alpha*np.eye(len(tr)); D=H[tr]; W=solve(G,np.column_stack((D,T[tr])),assume_a='pos'); n=D.shape[1]; b=np.linalg.lstsq(D.T@W[:,:n],D.T@W[:,n:],rcond=1e-10)[0]; return H[te]@b+K[np.ix_(te,tr)]@(W[:,n:]-W[:,:n]@b)

def variants(x):
 Q={'x':x,'A':tf(x,A),'Ai':(x-A[0])/np.diag(A[1:])}
 for k,v in list(Q.items()):
  Q[k+'B']=tf(v,B);Q[k+'Bi']=(v-B[0])@np.linalg.inv(B[1:]);Q[k+'Bt']=v@B[1:].T
 for k,v in list(Q.items()):
  if k in ('x','A','Ai','AiB','AB','xB'):
   Q[k+'C1']=c(v,'v1');Q[k+'C2']=c(v,'v2');Q[k+'BC1']=c(tf(v,B),'v1');Q[k+'BC2']=c(tf(v,B),'v2');Q[k+'C1B']=tf(Q[k+'C1'],B);Q[k+'C2B']=tf(Q[k+'C2'],B);Q[k+'Csum']=Q[k+'C1']+Q[k+'C2'];Q[k+'Cdiff']=Q[k+'C1']-Q[k+'C2'];Q[k+'Cprod']=Q[k+'C1']*Q[k+'C2']
 return Q

def features(q,kind,k):
 m={'sincos':[np.sin(k*q),np.cos(k*q)],'tanh':[np.tanh(k*q)],'ratio':[q/(1+(k*q)**2),1/(1+(k*q)**2)],'squash':[q/np.sqrt(1+(k*q)**2)],'warp':[q+k*q*q], 'raw':[]}[kind]
 return [q]+m

folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(26).permutation(72),4)]
if __name__=='__main__':
 Y=target; H0=base(x); H2=quad(x); raw=cdist(x,x,'sqeuclidean'); P0=np.exp(-raw/(2*1.35**2)); out=[]; predlist=[]
 for nm,q in variants(x).items():
  sd=q.std(0); qz=q/sd
  for typ,k in [('raw',1),('warp',-.18),('warp',.18)]+[(f,k) for f in ['sincos','tanh','ratio','squash'] for k in [.7,1.7,2.5]]:
   pack=features(q,typ,k); q1=pack[-1]; qn=q1/q1.std(0); F=np.column_stack(pack); F=(F-F.mean(0))/F.std(0); G=F@F.T / F.shape[1]; fq=qn@qn.T/4
   for kern,Kg in [('add',sum(np.exp(-cdist(a[:,None],a[:,None],'sqeuclidean')/2) for a in F.T)/F.shape[1]),('poly',(1+G)**2/4),('pair',(1+fq)**2/4),('rbf',np.exp(-cdist(qn,qn,'sqeuclidean')/8))]:
    for mix in [0,.2]:
     K=Kg+mix*P0
     for a in [.001,.02,.12]:
      pred=[fitpred(K,H0,Y,tr,te,a) for tr,te in folds]; es=np.array([abs(p-Y[te]).mean(0) for p,(_,te) in zip(pred,folds)]); out.append((es[:,0].max(),es[:,0].mean(),es[0,0],es[:,1:].mean(),nm,typ,k,kern,mix,a)); predlist.append(pred)
 for idx, title in [(0,'worst y'),(1,'mean y'),(2,'52 y'),(3,'mean traj')]:
  order=sorted(range(len(out)),key=lambda j:out[j][idx]); print(title,*[tuple(round(d,5) if isinstance(d,float) else d for d in out[j]) for j in order[:7]],sep='\n');
  for n in [8,30,150]:
   es=np.array([abs(np.mean([predlist[j][i] for j in order[:n]],0)-Y[te]).mean(0) for i,(_,te) in enumerate(folds)]);print(n,'ensemble y',es[:,0].round(5),'traj',es[:,1:].mean(1).round(5))
