import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=np.load('/tmp/probe.npz')['u']; n=len(z)
P=list(combinations(range(4),2)); T=list(combinations(range(4),3));
def ff(x,k,f):
 if f=='sin':return np.sin(k*x)
 if f=='cos':return np.cos(k*x)-1
 if f=='rat':return x/(1+(k*x)**2)
 return np.tanh(k*x)
def design(x,k,f,extra):
 return np.column_stack((np.ones(len(x)),x,ff(x,k[0],f),*([x*x] if extra=='square' else []),*[np.prod(x[:,p],axis=1) for p in (P if extra=='pair' else T if extra=='triple' else [])]))
D={}
for lab,key in [('b','v'),('c','v1'),('c','v2')]:
 X,Y=s['panel_'+lab]['u'],s['panel_'+lab][key]; results=[]
 for f in ('sin','cos','rat','tanh'):
  for extra in ('none','square','pair','triple'):
   def calc(k,ret=False):
    h=design(X,k,f,extra); b=np.linalg.lstsq(h,Y,rcond=None)[0]; r=h@b-Y
    if ret:
     q=np.linalg.qr(h,mode='reduced')[0]; return np.mean(abs(r/np.maximum(.01,1-(q*q).sum(1))[:,None])),np.mean(abs(r)),b
    return r.ravel()
   for start in (.8,1.5,2.3,3.):
    o=least_squares(calc,[start],bounds=(.25,4.5),max_nfev=60); l,r,b=calc(o.x,True); results.append((l,r,f,extra,o.x,b))
 results.sort(key=lambda t:t[0]); l,r,f,e,k,b=results[0]; print(lab+key,[(round(t[0],6),round(t[1],6),t[2],t[3],round(t[4][0],4)) for t in results[::2][:5]]); print(np.round(b,3)); D[lab+key]=design(z,k,f,e)@b
np.savez('/tmp/struct6.npz',**D)
y=s['train']['y']; tr=s['train']['traj']; _,sv,v=np.linalg.svd(tr-tr.mean(0),full_matrices=False); print('centered spectrum',sv,'modes',v[:2].round(3)); print('correlation',np.corrcoef(np.column_stack((z,y,tr@v[:2].T)).T).round(2)); print('target ranges',np.quantile(np.column_stack((y,tr@v[:2].T)),[0,.25,.5,.75,1],axis=0).round(3)); print('pairs',np.round(np.column_stack((z,y,tr@v[:2].T))[:12],3))
