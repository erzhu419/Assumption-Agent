import numpy as np,pickle
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.linalg.norm(u-v,axis=1)<1; b,d=np.array([np.linalg.lstsq(np.c_[np.ones(sum(keep)),u[keep,i]],v[keep,i],rcond=None)[0] for i in range(4)]).T
T=-np.expm1(-t[:,None]*[1.8,.4]); S=z@np.linalg.pinv(T.T); W=np.c_[y,S]
print('amplitude range',W.min(0),W.max(0),'rms noise',np.std(z-S@T.T)); print('corr',np.corrcoef(W.T)); print('extreme rows',np.c_[x,W][np.argsort(abs(y))[-12:]])
links={'lin':lambda a:a,'tanh':np.tanh,'sat':lambda a:a/np.sqrt(1+a*a),'atan':np.arctan,'rational':lambda a:a/(1+.22*a*a)}
def probe(X,W,n=52,pan=False):
 P=PolynomialFeatures(3).fit(X); F=P.transform(X); e=P.powers_; a=e.sum(1); q=(e>0).sum(1); names=P.get_feature_names_out(); cols=[np.ones(len(a),bool),a<=2,q<=1,(q<=1)|(a<=2),(q<=1)|(q==3), (q<=1)|(a<=2)|(q==3), (a<=1)|((a==3)&(q<=1))];
 for mode,k in enumerate(cols):
  if pan and k.sum()>=n-2:continue
  A=F[:,k]; C=np.linalg.lstsq(A[:n],W[:n],rcond=None)[0]; pred=A@C; print(mode,'fit',np.round(np.std(W[:n]-pred[:n],0),5),'test',np.round(np.mean(abs(W[n:]-pred[n:]),0),5));
  if mode in (2,3,6) and not pan:
   print('coefs',list(zip(names[k],np.round(C,4).tolist())))
  if mode in (2,3,4,5,6) and not pan:
   for j in range(W.shape[1]):
    fits=[]
    for name,link in links.items():
     if name=='lin':continue
     for offset in ([0,1] if j==0 else [0]):
      def f(p):return offset*p[-2]+p[-1]*link(A@p[:-2]);
      sc=max(abs(W[:n,j]))*1.1 if name!='rational' else 1.; p=np.r_[C[:,j]/sc,0.,sc]; o=least_squares(lambda p:f(p)[:n]-W[:n,j],p,max_nfev=170); fits.append((round(np.std(o.fun),5),round(np.mean(abs(f(o.x)[n:]-W[n:,j])),5),name,offset,round(o.x[-1],4)))
    print(j,sorted(fits)[:4])
for l,X in [('raw',x),('inverse',(x-b)/d),('forward',x*d+b)]:
 print(l); probe(X,W)
for name in ('panel_b','panel_c'):
 p=s[name];
 for k,v in p.items():
  if k!='u':
   print(name,k);probe(p['u'],v,19,True)
