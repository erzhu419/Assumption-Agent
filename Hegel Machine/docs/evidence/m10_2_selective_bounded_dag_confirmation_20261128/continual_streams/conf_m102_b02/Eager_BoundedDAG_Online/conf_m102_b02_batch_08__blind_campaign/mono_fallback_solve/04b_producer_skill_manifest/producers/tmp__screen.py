import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; g=t['traj']; x=t['x']; n=len(x)
D=lambda x:np.c_[np.ones(len(x)),x]
fit=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0]
Q=lambda x:np.c_[D(x),*[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2)]]
a=s['panel_a']; A=D(a['u']); av=a['v']; keep=np.ones(len(A),bool)
for _ in range(8):
 c=fit(A[keep],av[keep]); e=np.linalg.norm(A@c-av,axis=1); keep=e<3*np.median(e)
 print('cal trimmed',keep.sum(),np.median(e),np.sort(e)[-7:])
z=np.linalg.solve(c[1:].T,(x-c[0]).T).T; print('cal',c.round(4));
p=s['panel_c']; C1=fit(Q(p['u']),p['v1']); C2=fit(Q(p['u']),p['v2']); print('C1\n',C1.round(4),'\nC2\n',C2.round(4));
T=np.c_[y,g]; splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),i),i) for i in np.array_split(np.random.default_rng(88).permutation(n),4)]
features={}
for tag,w in [('raw',x),('cal',z)]:
 u=s['panel_b']['u']; v=s['panel_b']['v']; b=D(w)@fit(D(u),v); b2=Q(w)@fit(Q(u),v); c1=Q(w)@C1;c2=Q(w)@C2
 for k,h in [('z',w),('b',b),('b2',b2),('c1',c1),('c2',c2),('c1c2',np.c_[c1,c2]),('zc1',np.c_[w,c1]),('zc2',np.c_[w,c2]),('zc12',np.c_[w,c1,c2]),('zb2',np.c_[w,b2]),('all',np.c_[w,b2,c1,c2])]:features[tag+'_'+k]=h
scores=[[],[]]
for k,F in features.items():
 F=F/np.maximum(F[:52].std(0),1e-7); L=F@F.T/F.shape[1]; R=cdist(F,F,'sqeuclidean')/F.shape[1]; R1=abs(F[:,None,:]-F[None,:,:]);
 for mode in ('lin','quad','cubic','mat','rbf','add','add2','sat','sat2'):
  for scale in ([.45,1.,2.] if mode not in ('lin','quad','cubic') else [1.]):
   r=np.sqrt(R)*scale; d=R1*scale; M=((1+d+d*d/3)*np.exp(-d)).mean(2)
   K={'lin':1+L,'quad':(1+L)**2,'cubic':(1+L)**3,'mat':1+L+(1+r+r*r/3)*np.exp(-r),'rbf':1+L+np.exp(-r*r),'add':1+L+M,'add2':1+L+M+M*M,'sat':1+L+np.tanh(F*scale)@np.tanh(F*scale).T/F.shape[1],'sat2':(1+L+np.tanh(F*scale)@np.tanh(F*scale).T/F.shape[1])**2}[mode]
   for reg in (1e-6,1e-4,.003,.025):
    E=[]
    for i,j in splits:
     P=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),T[i]); e=np.abs(P-T[j]); E.append([e[:,0].mean(),e[:,1:].mean()]);
    E=np.array(E)
    for j in range(2):scores[j].append((E[:,j].mean(),E[:,j].max(),E[0,j],k,mode,scale,reg))
for typ,ss in zip(('scalar','state'),scores):
 print(typ,'mean/worst/52-20'); print(*[tuple(round(v,5) if isinstance(v,float) else v for v in r) for r in sorted(ss)[:10]],sep='\n'); print('best first'); print(*sorted(ss,key=lambda r:r[2])[:4],sep='\n')
