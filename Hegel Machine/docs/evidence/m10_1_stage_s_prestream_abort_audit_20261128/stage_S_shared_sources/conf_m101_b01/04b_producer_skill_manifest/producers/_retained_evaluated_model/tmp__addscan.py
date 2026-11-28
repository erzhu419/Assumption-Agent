exec(open('/tmp/discover.py').read().split('U,D,V=')[0])
from scipy.optimize import least_squares
from itertools import product,combinations
from scipy.linalg import orthogonal_procrustes
rng=np.random.default_rng(314); pc=s['panel_c']; u=pc['u']; Z=np.c_[pc['v1'],pc['v2']]; i=np.arange(len(u)); ii=i[i%4!=0]; jj=i[i%4==0]
fn={'s':np.sin,'c':np.cos,'t':np.tanh,'r':lambda a:a/(1+a*a),'p':lambda a:a*a,'o':lambda a:a**3}
def fit(U,Y,ind,seq,lin=True,start=2.,shift=False):
 def design(X,p):
  cols=[np.ones(len(X))];
  if lin:cols.extend(X.T)
  for k,typ in enumerate(seq):
   if typ!='-': cols.append(fn[typ](X[:,k%4]*p[k]+(p[len(seq)+k] if shift else 0)))
  return np.column_stack(cols)
 def get(p):
  F=design(U[ind],p); w=np.linalg.lstsq(F,Y[ind],rcond=None)[0];return F@w-Y[ind],w
 p0=np.full(len(seq),start);lo=np.full(len(seq),.15);hi=np.full(len(seq),6.)
 if shift:p0=np.r_[p0,np.zeros(len(seq))];lo=np.r_[lo,np.full(len(seq),-1.5)];hi=np.r_[hi,np.full(len(seq),1.5)]
 sol=least_squares(lambda p:get(p)[0].ravel(),p0,bounds=(lo,hi),max_nfev=100);e,w=get(sol.x);return design(U,sol.x)@w,np.mean(abs(e)),sol.x,w
for Z1 in [Z[:,:4],Z[:,4:]]:
 trials=[]
 for lin in [True,False]:
  for seq in ['ssss','cccc','tttt','rrrr','ttttssss','ttttcccc','rrrrcccc','sssscccc','ttttrrrr','pppp','oooo']:
   for shift in ([False,True] if len(seq)==4 else [False]):
    p,e,w,_=fit(u,Z1,ii,seq,lin,2.,shift); trials.append((np.mean(abs(p[jj]-Z1[jj])),e,lin,seq,shift,w.round(3)))
 print('C best',sorted(trials,key=lambda a:a[0])[:9])
q=coords['AinvB']; tt=np.c_[1-np.exp(-.365*t),1-np.exp(-1.9076*t)]; z=tr@np.linalg.pinv(tt.T); Y=np.c_[y,z]; i=np.arange(len(y)); ii=i[:56]; jj=i[56:]
for j in range(3):
 trials=[]
 for seq in ['ssss','tttt','rrrr','sssscccc','ttttcccc','rrrrcccc','ttttrrrr','ttttssss','rrrrssss','cccc','oooo','ssssssss'] + list(map(''.join,product('str',repeat=4))):
  for shift in ([False,True] if seq in ['ssss','tttt','rrrr','sssscccc','ttttcccc','rrrrcccc','ttttrrrr','ttttssss','rrrrssss','cccc'] else [False]):
   p,e,pr,w=fit(q,Y[:,j:j+1],ii,seq,True,2.,shift); trials.append((np.mean(abs(p[jj,0]-Y[jj,j])),e,seq,shift,pr.round(2),w.ravel().round(3)))
 print('TARGET',j,'best',sorted(trials,key=lambda a:a[0])[:5])
