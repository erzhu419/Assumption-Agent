import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for key in ('v1','v2'):
 p=S['panel_c']; x=p['u']; y=p[key]; lin=np.c_[np.ones(len(x)),x]; pairs=list(combinations(range(4),2)); a,b=x[:,[i for i,j in pairs]],x[:,[j for i,j in pairs]]
 kinds={'tanh':lambda k:np.tanh(k*x),'arctan':lambda k:np.arctan(k*x),'sine':lambda k:np.sin(k*x),'softsign':lambda k:x/(1+k*abs(x)),'rat':lambda k:x/(1+k*x*x),'sqrt':lambda k:x/np.sqrt(1+k*x*x),'cos':lambda k:np.cos(k*x),'exp':lambda k:np.exp(-k*x*x),'square_rat':lambda k:x*x/(1+k*x*x),'quad_cos':lambda k:np.cos(k*x*x),'cos_pair':lambda k:np.c_[np.cos(k*(a+b)),np.cos(k*(a-b))],'pair_rat':lambda k:np.c_[a*b/(1+k*(a*a+b*b))], 'pair_sumrat':lambda k:np.c_[(a+b)/(1+k*(a+b)**2),(a-b)/(1+k*(a-b)**2)]}
 for name,fn in kinds.items():
  def run(k):
   d=np.c_[lin,fn(k)]; c=np.linalg.lstsq(d,y,rcond=None)[0]; e=y-d@c; return np.mean(e**2), c
  r=minimize_scalar(lambda k:run(k)[0],bounds=(.005,3.5),method='bounded'); print(key,name,round(r.x,5),round(r.fun**.5,7));
  if r.fun<.00003: print(np.round(run(r.x)[1],5))
