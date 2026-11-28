from readout_probe import *
from scipy.spatial.distance import cdist
D=list(batches()); print('trajectory', [round(np.abs(pr-tr).mean(),5) for p,pr,y,tr in D])
def center(K,L):
 c=K.mean(0); return K-c-c[:,None]+c.mean(), L-c-L.mean(1,keepdims=True)+c.mean()
def setup(p,pr):
 K,L=p['K'],p['Kq']; C,M=center(K,L); e,U=np.linalg.eigh(C); U=U[:,-4:]; e=e[-4:]; A,B=U*np.sqrt(e), (M@U)/np.sqrt(e)
 out={'x':(A,B),'s':(p['sy'],p['qy']),'t':(p['st'][:,:4],p['qt'][:,:4]),'a':(p['scores'],pr@p['modes'].T)}
 terms={'k':(C,M)}
 for key,(x,q) in out.items():
  sd=x.std(0); x,q=(x-x.mean(0))/sd,(q-x.mean(0))/sd
  D,E=cdist(x,x,'sqeuclidean'),cdist(q,x,'sqeuclidean'); P,Q=x@x.T,q@x.T
  for deg in [1,2,3,4,5]: terms[key+str(deg)]=center((1+P/len(sd))**deg,(1+Q/len(sd))**deg)
  for w in [.5,1,2,4,8,16,32]: terms[key+'r'+str(w)]=center(np.exp(-D/w),np.exp(-E/w))
 return terms
V=[(setup(p,pr),p['y_mean'],p['y_centered'],y) for p,pr,y,tr in D]
def eval(combo,r):
 errors=[]
 for d,mu,y,z in V:
  K=sum(w*d[k][0] for k,w in combo); L=sum(w*d[k][1] for k,w in combo)
  errors.append(np.abs(mu+L@solve(K+r*np.eye(len(K)),y,assume_a='pos')-z).mean())
 return np.array(errors)
cs=[]
for k in V[0][0]:
 for w in [.05,.3,1,3] if k!='k' else [1]:
  for r in [.00003,.0003,.003,.03,.3]:
   combo=[('k',1),(k,w),('t2',.03),('a1',.03)]
   e=eval(combo,r);cs.append((e.max(),e.mean(),e[0],k,w,r,e))
for lo in [0,1,2]:
 print('ranking',lo)
 for c in sorted(cs,key=lambda c:c[lo])[:5]:print(c[:6],np.round(c[6],4))
p,pr,y,_=D[0];print('feature ranges',[(k,np.round(p[k].std(0),3)) for k in ['sy','st','scores']]); print('targets',np.round(y,3))
