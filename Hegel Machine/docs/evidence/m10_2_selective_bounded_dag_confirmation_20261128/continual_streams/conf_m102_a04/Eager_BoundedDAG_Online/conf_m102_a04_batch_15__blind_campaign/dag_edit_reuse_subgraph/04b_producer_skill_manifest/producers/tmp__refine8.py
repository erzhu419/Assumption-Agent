import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_c']['u'];y=s['panel_c']['v2'];p=np.roll(x,1,axis=1);t=np.roll(x,-1,axis=1)
def gen(x,m,q,r):
 p=np.roll(x,1,axis=1);t=np.roll(x,-1,axis=1)
 g={'prod':lambda:x*p,'sinprev':lambda:np.sin(r*x)*p/r,'sinnext':lambda:np.sin(r*x)*t/r,'sinboth':lambda:np.sin(r*np.roll(x,-1,axis=1))*x/r,'sinprod':lambda:np.sin(r*x*p)/r,'damp':lambda:x*p/(1+r*x*x),'dampother':lambda:x*p/(1+r*p*p),'sumdamp':lambda:x*p/(1+r*(x+p)**2),'diffdamp':lambda:x*p/(1+r*(x-p)**2)}[m]()
 return g
rows=[]
for m in ['prod','sinprev','sinnext','sinboth','sinprod','damp','dampother','sumdamp','diffdamp']:
 for mode in ['none','linear']:
  for basis in ['rational','tanh','sin']:
   def f(x,theta):
    q,r=theta; h={'rational':lambda:x/(1+q*x*x),'tanh':lambda:np.tanh(q*x)/q,'sin':lambda:np.sin(q*x)/q}[basis](); return np.column_stack([np.ones(len(x)),h,gen(x,m,q,r)]+([x] if mode=='linear' else []))
   def err(theta):
    a=f(x,theta);return (a@np.linalg.lstsq(a,y,rcond=1e-9)[0]-y).ravel()
   fits=[least_squares(err,[q,r],bounds=([.002,.002],[5,5]),gtol=1e-11) for q,r in [(.35,.3),(.6,.6),(1.3,1.3),(2.6,.4)]];o=min(fits,key=lambda a:a.cost);a=f(x,o.x);e=err(o.x).reshape(y.shape);h=(np.linalg.svd(a,full_matrices=False)[0]**2).sum(1);rm=np.sqrt(np.mean(e*e));cv=np.sqrt(np.mean((e/(1-h[:,None]))**2));bic=y.size*np.log(rm**2)+(a.shape[1]*y.shape[1]+1+(m!='prod'))*np.log(y.size);rows.append((bic,rm,cv,m,mode,basis,o.x,np.linalg.lstsq(a,y,rcond=1e-9)[0]))
for row in sorted(rows,key=lambda r:r[0])[:15]:
 print(row[:7]);
 if row is min(rows,key=lambda r:r[0]):print('coef',row[7])
