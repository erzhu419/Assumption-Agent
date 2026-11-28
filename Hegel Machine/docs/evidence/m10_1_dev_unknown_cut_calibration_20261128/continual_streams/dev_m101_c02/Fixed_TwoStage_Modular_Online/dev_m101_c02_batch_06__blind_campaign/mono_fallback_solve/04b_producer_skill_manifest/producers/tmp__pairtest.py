from discover import *
from itertools import combinations
tr=s['train']; X=cal(tr['x'],True); U=s['panel_b']['u']; V=s['panel_b']['v']; f=lambda x:np.c_[one(x),np.sin(2.8*x)]; Z=f(X)@fit(f(U),V); H=one(Z); Y=np.c_[tr['y'],tr['traj']]; N=len(X)
D={}; K={}; t=s['eval_times']; pairs=list(combinations(range(4),2))
for nm,W in [('X',X),('B',Z),('preB',X-.4*np.sin(2.8*X))]:
 for i,j in pairs:
  a,b=W[:,i],W[:,j]; w=np.c_[a,b];d=cdist(w/w.std(0),w/w.std(0),'sqeuclidean')/2
  K[nm+str(i)+str(j)]=[np.exp(-d/(2*l*l)) for l in [.22,.4,.7,1.2]]
  dct={'prod':a*b,'diff':a-b,'sum':a+b,'diff05':a-.5*b,'sum05':a+.5*b,'rev05':b-.5*a,'adiff':a*a-b*b,'ssq':a*a+b*b}
  dd={}
  for k,v in dct.items():
   dd[k]=v
   for c in [.7,1.4,2.1,2.8,4.2,5.6]:
    for name,g in [('sin',np.sin),('cos',np.cos),('atan',np.arctan),('tanh',np.tanh),('rat',lambda z:z/(1+z*z))]:dd[k+name+str(c)]=g(c*v)
  for l in [1,2.1,2.8,4.2]:
   for n,g in [('sin',np.sin),('cos',np.cos),('atan',np.arctan),('sq',lambda z:z*z),('id',lambda z:z)]:
    for m,h in [('sin',np.sin),('cos',np.cos),('atan',np.arctan),('sq',lambda z:z*z),('id',lambda z:z)]:dd[n+m+str(l)]=g(l*a)*h(l*b)
  for k,v in dd.items():D[nm+str(i)+str(j)+'_'+k]=v
keys=list(D); F=np.array(list(D.values())).T
sp=[np.arange(N)]+[np.random.default_rng(k).permutation(N) for k in range(5)]
def stats(pred, b):
 er=abs(pred-Y[b]);return np.r_[er[:,0].mean(),er[:,1:].mean(),er[:,-1].mean()]
def kr(K,a,b,r):
 h=H[a];M=np.block([[K[np.ix_(a,a)]+r*np.eye(len(a)),h],[h.T,np.zeros((5,5))]]);Mi=np.linalg.inv(M);c=Mi@np.r_[Y[a],np.zeros((5,9))];e=abs(c[:len(a)]/Mi.diagonal()[:len(a),None]);return e[:,0].mean()+1.7*e[:,1:].mean(),stats(np.c_[K[np.ix_(b,a)],H[b]]@c,b)
sc={i:[] for i in range(1,6)}; sk={k:[] for k in ['individual','allX','top2X','combo']}
for ids in sp:
 a,b=ids[:56],ids[56:]; h=H
 for k in sc:
  r=Y[a]-h[a]@fit(h[a],Y[a]); v=np.linalg.svd(r,full_matrices=False)[2][0]; Q=np.linalg.qr(h[a])[0]; p=F[a]-Q@(Q.T@F[a]); score=abs(p.T@(r@v))/np.maximum(np.linalg.norm(p,axis=0),1e-8); j=score.argmax();h=np.c_[h,F[:,j]];sc[k].append(stats(h[b]@fit(h[a],Y[a]),b));print('term',k,keys[j],np.linalg.norm(Y[a]-h[a]@fit(h[a],Y[a])).round(4))
 cand={n:min((kr(L,a,b,r)[0],l,r) for l,L in enumerate(Ls) for r in [.003,.03,.3,3.]) for n,Ls in K.items()}; ranking=sorted(cand,key=lambda n:cand[n][0]); print('pairs',[(n,tuple(round(x,4) for x in cand[n])) for n in ranking[:6]]);
 xrank=[n for n in ranking if n[0]=='X']; LS={ 'individual':K[ranking[0]], 'allX':[np.mean([K[n][l] for n in xrank],axis=0) for l in range(4)], 'top2X':[np.mean([K[n][l] for n in xrank[:2]],axis=0) for l in range(4)]}; LS['combo']=LS['top2X']+LS['allX']+LS['individual']
 for key,Ls in LS.items():sk[key].append(min([kr(L,a,b,r) for L in Ls for r in [.003,.03,.3,3.]],key=lambda p:p[0])[1])
for k,e in (sc|sk).items():print(k,'first, mean, max',np.round([e[0],np.mean(e,axis=0),np.max(e,axis=0)],5).tolist())
