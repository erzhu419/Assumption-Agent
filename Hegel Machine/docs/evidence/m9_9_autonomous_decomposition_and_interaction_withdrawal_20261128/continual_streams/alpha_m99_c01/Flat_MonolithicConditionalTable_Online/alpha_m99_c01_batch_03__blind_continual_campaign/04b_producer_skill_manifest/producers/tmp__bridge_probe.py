import pickle,numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; R=C['bridge_raw']; Y=C['true_channel_loads']; n=len(Y)
np.set_printoptions(precision=5,suppress=True,linewidth=140)
print('cal rows',n,'true-channel max',np.max(np.abs(Y),axis=0)); print('extreme samples [index,true channels,raw channels]'); i=np.argsort(np.max(np.abs(R),axis=1))[-12:]; print(np.c_[i,Y[i],R[i]])
print('largest deployment raw component by split',[(k,f,np.max(np.abs(D[k][f]))) for k in ('train','test') for f in ('past_channel_raw','hold_channel_raw')])
terms=[()]+[(i,) for i in range(4)]+list(combinations_with_replacement(range(4),2))+list(combinations_with_replacement(range(4),3))
def des(terms): return np.column_stack([np.prod(Y[:,ix],axis=1) if ix else np.ones(n) for ix in terms])
fold=np.random.default_rng(17).permutation(n)%5
for cutoff in [3.5,5.,10.,100.]:
 m=np.max(np.abs(R),axis=1)<cutoff
 for name,T in [('linear',terms[:5]),('compact cubic',[t for t in terms if len(t)!=3 or len(set(t))==1]),('full cubic',terms)]:
  A=des(T); coef=np.linalg.lstsq(A[m],R[m],rcond=None)[0]; E=np.empty_like(R)
  for j in range(5):
   tr=m&(fold!=j); test=fold==j; E[test]=R[test]-A[test]@np.linalg.lstsq(A[tr],R[tr],rcond=None)[0]
  print(cutoff,name,m.sum(),'channel RMSE nominal',np.sqrt(np.mean(E[m]**2,axis=0)),'worst nominal fold',max(np.sqrt(np.mean(E[m&(fold==j)]**2)) for j in range(5)),'largest excluded',np.max(np.abs(E[~m]),initial=0))
  if cutoff==5. and name=='compact cubic':
   print('terms and forward coefficients',list(zip(T,coef.round(5).tolist()))); print('excluded residuals [true, res]',np.c_[Y[~m],(R-A@coef)[~m]])
