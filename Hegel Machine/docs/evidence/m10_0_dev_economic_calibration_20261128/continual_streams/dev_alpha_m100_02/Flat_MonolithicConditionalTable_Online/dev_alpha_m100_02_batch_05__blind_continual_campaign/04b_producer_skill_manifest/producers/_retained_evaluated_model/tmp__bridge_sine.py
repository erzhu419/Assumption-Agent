import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; S=C['single_channel_sweeps']; clean=(abs(Y)<5).all(1); sw=abs(S['cmd_stress'])<5
j=S['lab_id'][sw]; z=S['true_stress'][sw]; v=S['cmd_stress'][sw]
def axis(k):
 a=np.stack([np.ones_like(z),z,np.sin(k*z)],1); b=np.stack([np.linalg.lstsq(a[j==q],v[j==q],rcond=None)[0] for q in range(4)])
 return np.sum(a*b[j],1)-v,b
k=least_squares(lambda p:axis(p[0])[0],[2.5]).x[0]; b=axis(k)[1]; print('frequency',k,'coeffs',b)
folds=np.array_split(np.random.default_rng(42).permutation(np.flatnonzero(clean)),5)
for q in range(4):
 x=X[:,q]; w=X[:,(q+2)%4]; u=X[:,(q+1)%4]; h=X[:,(q+3)%4]; target=Y[:,q]-(b[q,0]+b[q,1]*x+b[q,2]*np.sin(k*x));
 terms={'w':w,'w2':w*w,'w3':w**3,'sin(w)':np.sin(k*w),'atan(w)':np.arctan(w),'xw':x*w,'xu':x*u,'xh':x*h,'uw':u*w,'hw':h*w,'uh':u*h,'x2w':x*x*w,'xw2':x*w*w,'sin(x)w':np.sin(k*x)*w,'xsin(w)':x*np.sin(k*w),'w/(1+x2)':w/(1+x*x),'xw/(1+w2)':x*w/(1+w*w),'w/(1+w2)':w/(1+w*w),'w2/(1+w2)':w*w/(1+w*w),'xw/(1+x2)':x*w/(1+x*x),'u':u,'h':h}
 print('channel',q,'axis cross',np.column_stack([w,target])[(abs(w)>0)&(x==0)&clean]);
 choices=[['w'],['w','w2'],['w','w2','w3'],['w','sin(w)','w2'],['w','atan(w)','w2'],['w','w/(1+w2)','w2'],['w','w/(1+w2)','w2/(1+w2)'],['w','w2','xw'],['w','w2','w3','xw','xu','xh','uw','hw','uh'],list(terms)]
 for names in choices:
  a=np.column_stack([terms[n] for n in names]); e=np.zeros(len(X))
  for ids in folds:
   tr=clean.copy(); tr[ids]=False; c=np.linalg.lstsq(a[tr],target[tr],rcond=None)[0]; e[ids]=a[ids]@c-target[ids]
  c=np.linalg.lstsq(a[clean],target[clean],rcond=None)[0]; print(','.join(names),np.quantile(abs(e[clean]),[.5,.9,1]).round(6),c.round(5) if len(names)<12 else '')
 sel=[]
 for _ in range(8):
  def fit(names):
   aa=np.column_stack([terms[n] for n in names])[clean]; cc=np.linalg.lstsq(aa,target[clean],rcond=None)[0]; return np.linalg.norm(aa@cc-target[clean]),cc
  sel.append(min(set(terms)-set(sel),key=lambda n:fit(sel+[n])[0])); print('greedy',sel,fit(sel))
