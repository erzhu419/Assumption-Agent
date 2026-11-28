import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def aff(x): return np.c_[np.ones(len(x)),x]
def solve(f,y): return np.linalg.lstsq(f,y,rcond=None)[0]
def pp(x):
 i,j=np.triu_indices(4);return np.c_[aff(x),x[:,i]*x[:,j]]
if __name__=='__main__':
 print('time telemetry summary',s['eval_times'],s['telemetry_vec'],s['train']['summary'])
 for key in ['panel_a','panel_c']:
  for name,Y in s[key].items():
   if name=='u':continue
   x=s[key]['u']; k=np.linalg.norm(Y,axis=1)<8; x,y=x[k],Y[k]; F=pp(x); w=solve(F,y); print(key,name,'fit and rmse',w,np.std(F@w-y,axis=0))
   r=y-aff(x)@solve(aff(x),y)
   for j in range(4):
    rows=[]
    for l in range(4):
     for nm,g in [('sq',x[:,l]**2),('cube',x[:,l]**3),('tanh',np.tanh(x[:,l])),('sin',np.sin(x[:,l])),('sat',x[:,l]/(1+x[:,l]**2)),('gauss',x[:,l]*np.exp(-x[:,l]**2)),('cross',x[:,l]*x[:,(l+1)%4])]:
      g=g-aff(x)@solve(aff(x),g); rows.append((round(abs(np.corrcoef(g,r[:,j])[0,1]),3),l,nm))
    print(j,sorted(rows,reverse=True)[:9])
 x=s['train']['x']; t=s['eval_times']; B=-np.expm1(-t[:,None]*[1.8,.4]); tar=np.c_[s['train']['y'],solve(B,s['train']['traj'].T).T]; print('targets',tar[:8]); w=solve(pp(x)[:52],tar[:52]);print('target quad coefficients',w)
