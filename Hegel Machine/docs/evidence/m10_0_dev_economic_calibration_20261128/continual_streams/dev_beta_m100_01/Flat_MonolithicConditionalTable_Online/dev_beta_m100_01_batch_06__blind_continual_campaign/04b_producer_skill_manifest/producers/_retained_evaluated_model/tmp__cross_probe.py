import numpy as np,pickle,json
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; C=s['sensor_calibration_sweeps']; X,Y=C['ref_rosette'],C['raw_rosette']; keep=np.max(abs(Y),axis=1)<4; X,Y=X[keep],Y[keep]; ax=np.count_nonzero(X,axis=1)==1; q=C['single_bank_sweeps']; R=[]; A=[]; B=[]; O=[]
for j in range(4):
 a,b=X[:,j],X[:,(j+2)%4]; m=ax&(a!=0); n=(q['bank_id']==j)&(abs(q['raw_readout'])<4); t=np.r_[a[m],q['ref_signal'][n]]; z=np.r_[Y[m,j],q['raw_readout'][n]]; p=least_squares(lambda p:p[0]+p[1]*t+p[2]*np.sin(p[3]*t)-z,[0,1,.25,2.5]).x
 R.append((Y[:,j]-p[0]-p[1]*a-p[2]*np.sin(p[3]*a))/p[1]); A.append(a); B.append(b); O.append(p); print(j,p)
a,b,r=map(np.concatenate,(A,B,R)); group=np.tile(np.arange(len(X)),4); print('own dependence bins'); ids=np.argsort(b); print(np.c_[a,b,r][ids][::12].round(5))
for name in ['sin_cos','tanh_cos','tanh_rat','sin_rat','plain_rat','sincos_no_lin','tanh_quad','sin_gap','poly7','saturated_pair','tanh_gap']:
 def design(p):
  f,g=np.exp(p); L={ 'sin_cos':[b,np.sin(f*b),1-np.cos(g*b)], 'tanh_cos':[b,np.tanh(f*b),1-np.cos(g*b)], 'tanh_rat':[b,np.tanh(f*b),b*b/(1+g*b*b)],'sin_rat':[b,np.sin(f*b),b*b/(1+g*b*b)],'plain_rat':[b,b/(1+f*b*b),b*b/(1+g*b*b)],'sincos_no_lin':[np.sin(f*b),1-np.cos(g*b)],'tanh_quad':[b,np.tanh(f*b),b*b], 'sin_gap':[b,np.sin(f*b),a*np.sin(g*b),1-np.cos(g*b)],'poly7':[b**k for k in range(1,8)],'saturated_pair':[b,np.sin(2.5*(a+f*b))-np.sin(2.5*a),b*b/(1+g*b*b)],'tanh_gap':[b,np.tanh(f*b),np.tanh(g*b)**2]}[name];return np.array(L).T
 def fit(m):
  def pred(p,ret=False):
   D=design(p); c=np.linalg.lstsq(D[m],r[m],rcond=None)[0]; return (D@c-r,c) if ret else (D@c-r)[m]
  est=min((least_squares(pred,np.log([x,y]),bounds=(-4,1.7),max_nfev=100) for x,y in [(1,1),(2.5,2.5),(.4,2),(2,.4)]),key=lambda z:np.linalg.norm(z.fun)); return *pred(est.x,True),np.exp(est.x)
 e,c,p=fit(np.ones(len(a),bool)); cv=[]
 for f in range(4):
  m=group%4!=f; cv.append(np.sqrt(np.mean(fit(m)[0][~m]**2)))
 print(name,'max',max(abs(e)),'worst',max(cv),'p',p,'coef',c)
L=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for e in L:
 t=np.asarray(e['facility_telemetry_trace']); print(e['completed_campaign_short_name'], 'mean',t.mean(0).round(4),'tail',t[-10:].mean(0).round(4),'std',t.std(0).round(4))
