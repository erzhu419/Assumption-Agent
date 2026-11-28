exec(open('/tmp/probe.py').read().split('for k in (')[0])
from scipy.linalg import qr
from itertools import combinations
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; ids=ix if k=='panel_a' else np.arange(len(p['u'])); u=p['u'][ids];
 mon=[i for n in range(4) for i in combinations_with_replacement(range(4),n)]; D=pol(u,3); a=D[:,:5]; Q=qr(a,mode='economic')[0]
 funcs=[('poly',[i for i in mon[5:]],D[:,5:])]
 for name,fun in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('local',lambda x:x*np.exp(-x*x)),('rat',lambda x:x/(1+x*x))]:
  for freq in (.5,1,1.5,2,3):
   arr=[freq*u[:,j] for j in range(4)]+[freq*(u[:,i]+sign*u[:,j]) for i,j in combinations(range(4),2) for sign in (1,-1)]+[freq*u[:,i]*u[:,j] for i,j in combinations(range(4),2)]
   funcs.append((name,[(freq,j) for j in range(len(arr))],fun(np.array(arr).T)))
 for key,v in list(p.items())[1:]:
  y=v[ids]; print(k,key)
  if k=='panel_a':
   co=fit(D,y)
   for j in range(4):print(j,[(mon[i],round(co[i,j],4)) for i in np.argsort(abs(co[:,j]))[-10:][::-1]])
  scores=[]
  for name,labels,F in funcs:
   rem=F-Q@(Q.T@F); rem=rem/np.maximum(np.linalg.norm(rem,axis=0),1e-9)
   for out in range(4):
    li=[]; A=a
    for step in range(5):
     r=y[:,out]-A@fit(A,y[:,out]); i=np.argmax(abs(rem.T@r)); li.append(i); A=np.c_[a,F[:,li]]; U=qr(A,mode='economic')[0]; rem2=F-U@(U.T@F); rem=rem2/np.maximum(np.linalg.norm(rem2,axis=0),1e-9); rem[:,li]=0
     if step in (0,1,3,4):
      rr=y[:,out]-A@fit(A,y[:,out]); loo=np.mean(abs(rr/np.maximum(1-(U*U).sum(axis=1),1e-8))); scores.append((out,step+1,loo,name,[labels[j] for j in li]))
    rem=F-Q@(Q.T@F); rem=rem/np.maximum(np.linalg.norm(rem,axis=0),1e-9)
  for j in range(4): print(j,sorted([r[1:] for r in scores if r[0]==j],key=lambda r:r[1])[:4])
