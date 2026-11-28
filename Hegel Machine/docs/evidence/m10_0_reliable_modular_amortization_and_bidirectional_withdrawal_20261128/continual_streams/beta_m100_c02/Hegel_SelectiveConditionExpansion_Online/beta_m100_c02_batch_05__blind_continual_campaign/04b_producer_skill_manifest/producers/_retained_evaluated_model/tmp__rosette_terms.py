import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from itertools import combinations
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; d=c['single_bank_sweeps']; z=d['ref_signal'].reshape(4,-1).T; w=d['raw_readout'].reshape(4,-1).T; qa=abs(w)<4; good=np.max(abs(y),axis=1)<4
pairs=list(combinations(range(4),2)); folds=np.random.default_rng(20).permutation(len(x))%5
np.set_printoptions(precision=4,suppress=True)
print('axis normalized (banks 1-3):', np.c_[z[:,1],(w[:,1:]-w[5,1:])/(w[11,1:]-w[5,1:])]);
func={'tanh':lambda h,k:np.tanh(k*h),'root':lambda h,k:h/np.sqrt(1+k*h*h),'exp':lambda h,k:h*np.exp(-k*h*h),'rat':lambda h,k:h/(1+k*h*h),'erf':lambda h,k:erf(k*h)}
even={'sq':lambda h:h*h,'ratsq':lambda h:h*h/(1+h*h),'expsq':lambda h:-np.expm1(-h*h),'zero':lambda h:np.zeros_like(h)}
for kind,f in func.items():
 for ek,e in even.items():
  def resid(p):
   h=f(z,p[0]); out=[]
   for j in range(4):
    q=qa[:,j]; a=np.column_stack([np.ones(len(z)),h[:,j],z[:,j],e(z[:,j])]); t=np.linalg.lstsq(a[q],w[q,j],rcond=None)[0]; out.extend((a@t-w[:,j])[q]);
   return np.array(out)
  p=least_squares(resid,[1.],bounds=(.05,8)).x; print('axis',kind,ek,round(p[0],5),round(np.sqrt(np.mean(resid(p)**2)),8))

def design(v,k,kind,ek,interaction):
 h=func[kind](v,k); a=[np.ones(len(v)),v,h,even[ek](v)]
 if interaction=='lin': a += [np.column_stack([v[:,i]*v[:,j] for i,j in pairs])]
 if interaction=='sat': a += [np.column_stack([h[:,i]*h[:,j] for i,j in pairs])]
 if interaction=='rational': a += [np.column_stack([v[:,i]*v[:,j]/(1+v[:,i]**2+v[:,j]**2) for i,j in pairs])]
 return np.column_stack(a)
def fit(q,kind,ek,interaction):
 def f(k):
  a=design(x[q],k,kind,ek,interaction); return (a@np.linalg.lstsq(a,y[q],rcond=None)[0]-y[q]).ravel()
 k=least_squares(f,[1.],bounds=(.05,8)).x; a=design(x[q],k,kind,ek,interaction); return k,np.linalg.lstsq(a,y[q],rcond=None)[0]
scores=[]
for kind in func:
 for ek in ['sq','ratsq','expsq']:
  for interaction in ['none','lin','sat','rational']:
   o=[]
   for j in range(5):
    q=good&(folds==j); k,b=fit(good&~q,kind,ek,interaction); r=np.linalg.norm(design(x[q],k,kind,ek,interaction)@b-y[q],axis=1); o.append((np.median(r),max(r)))
   scores.append((max(t[0] for t in o),max(t[1] for t in o),kind,ek,interaction))
for row in sorted(scores)[:14]: print('cv',row)
for _,_,kind,ek,interaction in sorted(scores)[:3]:
 k,b=fit(good,kind,ek,interaction);print('coeff',kind,ek,interaction,k,b)
