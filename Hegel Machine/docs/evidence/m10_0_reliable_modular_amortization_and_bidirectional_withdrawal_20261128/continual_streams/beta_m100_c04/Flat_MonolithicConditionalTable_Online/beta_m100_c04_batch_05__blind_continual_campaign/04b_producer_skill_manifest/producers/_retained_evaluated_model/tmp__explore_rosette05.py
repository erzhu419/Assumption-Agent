import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; k=b['bank_id']; t=b['ref_signal']; z=b['raw_readout']; j,v=np.where((x!=0)&((x!=0).sum(1)==1)[:,None]); k=np.r_[k,v]; t=np.r_[t,x[j,v]]; z=np.r_[z,y[j,v]]; m=abs(z)<8; k,t,z=k[m],t[m],z[m]; single=(x!=0).sum(1)==1; clean=(abs(y)<8).all(1); bx=x[single&clean]; by=y[single&clean];
bias=np.array([np.median(by[(bx[:,a]==0)&(bx[:,(a+2)%4]==0),a]) for a in range(4)]); print('bias',bias)
forms={
 'atan+tanh':(lambda t,a,b,d:np.arctan(a*t)+b*np.tanh(d*t),[1.,1.,1.]),
 'tanh+linear':(lambda t,a,b:np.tanh(a*t)+b*t,[1.,.2]),
 'tanh+rational':(lambda t,a,b,d:np.tanh(a*t)+b*t/(1+d*t*t),[1.3,.15,.5]),
 'tanh+cubic':(lambda t,a,b,d:np.tanh(a*t)+b*t+d*t**3,[1.3,.2,0.]),
 'erf+cubic':(lambda t,a,b,d:erf(a*t)+b*t+d*t**3,[1.3,.2,0.]),
 'atan+cubic':(lambda t,a,b,d:np.arctan(a*t)+b*t+d*t**3,[1.3,.2,0.]),
 'tanh_cubicarg+linear':(lambda t,a,b,d:np.tanh(a*t+b*t**3)+d*t,[1.3,0,.15]),
 'sinhcosh':(lambda t,a,b:np.sinh(a*t)/(np.cosh(a*t)+b),[1.,.5]),
 'sinhcosh+linear':(lambda t,a,b,d:np.sinh(a*t)/(np.cosh(a*t)+b)+d*t,[1.,.5,.2]),
 'rational34':(lambda t,a,b,d:t*(1+a*t*t)/(1+b*t*t+d*t**4),[.25,.8,.05]),
 'rational56':(lambda t,a,b,d,e,f:t*(1+a*t*t+b*t**4)/(1+d*t*t+e*t**4+f*t**6),[.25,.02,.8,.05,.005]),
 'algebraic+linear':(lambda t,a,b:t/np.sqrt(1+a*t*t)+b*t,[1.,.2]),
 'odd_poly9':(lambda t,a,b,d,e:t*(1+a*t*t+b*t**4+d*t**6+e*t**8),[-.4,.1,-.02,.001]),
}
results=[]
for name,(f,start) in forms.items():
 def pred(p): return bias[k]+p[:4][k]*f(t,*p[4:]);
 def fit(mask): return least_squares(lambda p:(pred(p)-z)[mask],np.r_[np.ones(4),start],gtol=1e-12,max_nfev=350).x
 p=fit(np.ones(len(t),bool)); es=[]
 for val in np.unique(t):
  hold=t==val; es.extend((pred(fit(~hold))-z)[hold]);
 rms=np.sqrt(np.mean((pred(p)-z)**2)); print(name,p.round(6), 'fit',rms,'held',np.sqrt(np.mean(np.square(es))),np.max(np.abs(es)))
 results.append((rms,name,p,f))
for _,name,p,f in sorted(results)[:3]:
 print('\nCROSS',name)
 X=x[clean]; Y=y[clean]; F=f(X,*p[4:]); rest=Y-bias-p[:4]*F
 for a in range(4):
  g=bx[:,(a+2)%4]!=0
  print('isolated cross',a,np.c_[bx[g,(a+2)%4],by[g,a]-bias[a]].round(6))
  ix=[a,(a+1)%4,(a+2)%4,(a+3)%4]
  for deg in [1,2,3]:
   for src in ['raw','sat','both']:
    B={'raw':X[:,ix],'sat':F[:,ix],'both':np.c_[X[:,ix],F[:,ix]]}[src]; names=['v','next','opp','prev'] if src!='both' else ['v','next','opp','prev','satV','satN','satO','satP']; design=[]; terms=[]
    for d in range(1,deg+1):
     for q in combinations_with_replacement(range(B.shape[1]),d):
      if all(i in (0,4) for i in q):continue
      design.append(np.prod(B[:,q],axis=1)); terms.append('*'.join(names[i] for i in q))
    D=np.array(design).T; inds=np.arange(D.shape[1]); A=rest[:,a]; best=[]
    for size in range(min(9,len(terms))):
     opt=min((np.linalg.norm(D[:,best+[i]]@np.linalg.lstsq(D[:,best+[i]],A,rcond=None)[0]-A),i) for i in inds if i not in best)[1];best.append(opt)
    G=D[:,best]; coef=np.linalg.lstsq(G,A,rcond=None)[0]; err=A-G@coef; hat=np.sum(np.linalg.qr(G)[0]**2,axis=1); print(a,deg,src,round(np.std(err),6),round(np.max(abs(err/np.maximum(1-hat,1e-12))),6),[(terms[i],round(v,4)) for i,v in zip(best[:5],coef[:5])])
