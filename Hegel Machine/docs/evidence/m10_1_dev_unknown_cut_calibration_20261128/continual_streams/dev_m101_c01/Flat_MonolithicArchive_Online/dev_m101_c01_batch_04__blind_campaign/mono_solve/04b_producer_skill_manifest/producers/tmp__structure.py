exec(open('/tmp/probe.py').read().split('for p,k')[0])
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from itertools import combinations_with_replacement as cwr
pa=s['panel_a']; good=np.max(abs(pa['v']),axis=1)<2
ca=fit(poly(pa['u'][good],1),pa['v'][good]); bias=ca[0]; scale=ca[1:].diagonal()
pb=s['panel_b']; cb=fit(poly(pb['u'],1),pb['v']); pc=s['panel_c']; C=[]
for k in ['v1','v2']:
 print('\nPANEL C',k)
 for d in [1,2]:
  a=poly(pc['u'],d); c=fit(a,pc[k]); h=np.sum(a*np.linalg.pinv(a).T,axis=1); res=a@c-pc[k]; print(d,'rms',rm(res),'loo',rm(res/(1-h[:,None])))
  if d==2: print(list(zip([i for j in range(d+1) for i in cwr(range(4),j)],c.round(5).tolist()))); C.append(c)
print('true summary',y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std());print('z col means', z.mean(0)); print('t',t)
folds=[(np.arange(56),np.arange(56,72)),*KFold(5,shuffle=True,random_state=12).split(x)]
def score(K,Y,noise):
 es=[]
 for i,j in folds:
  r=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+noise*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]; es.append(np.mean(abs(r)))
 return np.round([es[0],np.mean(es[1:]),max(es[1:])],5)
P=poly(x,2); K=P@P.T
Q=poly(x,3); L=Q[:,15:]@Q[:,15:].T
for name,u in [('raw',x),('a_forward',x*scale+bias),('a_inverse',(x-bias)/scale)]:
 b=poly(u,1)@cb; c1,c2=[poly(u,2)@cc for cc in C]
 for nn,v in [('u',u),('b',b),('c1',c1),('c2',c2),('both_c',np.c_[c1,c2]),('all',np.c_[b,c1,c2])]:
  f=poly(v,2); G=K+f@f.T; D=cdist(v,v,'sqeuclidean')/v.shape[1]*4
  print(name,nn,[(l,score(G+np.exp(-D/(2*l*l)),y,.001).tolist(),score(G+np.exp(-D/(2*l*l)),z,.0001).tolist()) for l in [1.,2.,4.]])
for la in [.01,.1,1]:
 print('cubic',la,score(K+la*L,y,.003),score(K+la*L,z,.0001))
print('quadratic coefficients y/z_end',fit(P,np.c_[y,z[:,-1]]).round(5))
# Estimate which time modes are explainable by input, not just the low singular values of measured data.
for d in [1,2,3]:
 a=poly(x,d); p=a@fit(a,z); _,sv,h=np.linalg.svd(p,full_matrices=False); print('z predictable',d,sv.round(5),h[:4].round(4))
