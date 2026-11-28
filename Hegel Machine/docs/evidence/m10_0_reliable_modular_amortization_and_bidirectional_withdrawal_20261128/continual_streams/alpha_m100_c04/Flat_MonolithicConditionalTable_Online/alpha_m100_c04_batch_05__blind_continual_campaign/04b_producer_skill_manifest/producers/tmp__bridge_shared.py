import pickle,numpy as np
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; q=c['true_channel_loads']; r=c['bridge_raw']; good=np.max(abs(r),axis=1)<8; ax=(q!=0).sum(1)==1
v=np.roll(q,2,axis=1)
forms={'exp':lambda v,k:-np.expm1(-k*v*v),'tanh2':lambda v,k:np.tanh(k*v)**2,'cos':lambda v,k:1-np.cos(k*v),'tanhq2':lambda v,k:np.tanh(k*v*v),'rational':lambda v,k:v*v/(1+k*v*v),'softsqrt':lambda v,k:np.sqrt(1+k*v*v)-1,'stretchedexp':lambda v,k:-np.expm1(-k*v*v),'logcosh':lambda v,k:np.logaddexp(k*v,-k*v)-np.log(2)}
for name,E in forms.items():
 for extra in ('none','odd','even','power','mixed') if name=='exp' else ('none',):
  def pred(p):
   f,a,k,e,h,j=p[:6]; own=q+a*np.sin(f*q); zz=E(v,k) if extra!='power' else -np.expm1(-k*abs(v)**(2+h)); opp=v-e*zz
   if extra=='odd': opp+=h*(np.sin(j*v)-j*v)
   if extra=='even': opp+=h*v*v
   if extra=='mixed': opp+=h*np.roll(q,1,axis=1)*q
   return p[6:10]+p[10:14]*own+p[14:18]*opp
  def fit(m):
   p=np.r_[2.5,.25,1,.6,0.,1.7,0,0,0,0,1,1,1,1,.2,.2,.2,.2]; lo=np.r_[2,.1,.02,0,-.8,.2,np.full(12,-np.inf)]; hi=np.r_[3,.35,10,5,1.,5,np.full(12,np.inf)]
   return least_squares(lambda p:(pred(p)-r)[m].ravel(),p,bounds=(lo,hi),max_nfev=100).x
  p=fit(good); e=pred(p)-r; groups=[good&ax,good&~ax]; held=[]
  for k2 in range(5):
   m=good & (np.arange(len(good))%5==k2); held.extend((pred(fit(good&~m))-r)[m].ravel())
  print(name,extra,'params',p.round(6),'rms axis/mixed',*[np.sqrt(np.mean(e[m]**2)) for m in groups],'CV rms/max',np.sqrt(np.mean(np.square(held))),np.max(np.abs(held)))
  if name=='exp' and extra=='none':
   inds=np.argsort(np.max(abs(e[good]),axis=1))[-8:]; print('worst signed',np.c_[q[good][inds],e[good][inds]].round(5))
