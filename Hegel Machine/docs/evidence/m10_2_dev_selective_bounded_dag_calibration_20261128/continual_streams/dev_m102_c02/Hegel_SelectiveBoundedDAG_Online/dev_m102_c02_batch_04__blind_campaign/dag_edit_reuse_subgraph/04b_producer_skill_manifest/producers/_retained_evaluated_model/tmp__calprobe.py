import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=140)
def aff(x):return np.column_stack([np.ones(len(x)),x])
def regress(d,v):
 w=np.linalg.lstsq(d,v,rcond=None)[0]; return w,v-d@w
u,v=[s['panel_a'][k] for k in ('u','v')]
for typ, d in [('linear',aff(u)),('diag2',np.column_stack([aff(u),u*u])),('diag3',np.column_stack([aff(u),u**3])),('full2',np.column_stack([aff(u),*[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]])),('lin+sin',np.column_stack([aff(u),np.sin(u)])),('lin+tanh',np.column_stack([aff(u),np.tanh(u)]))]:
 w,r=regress(d,v);keep=np.ones(len(d),bool)
 for _ in range(12):
  w=least_squares(lambda z:(d@z.reshape(w.shape)-v).ravel(),w.ravel(),loss='cauchy',f_scale=.012).x.reshape(w.shape);break
 err=np.sqrt(np.mean((d@w-v)**2,axis=1));print('A',typ,'med',np.median(err),'sorted',np.sort(err), 'coef',w if typ in ('linear','full2','diag3') else '')
u,v=[s['panel_b'][k] for k in ('u','v')];print('B extrema',u.min(0),u.max(0),v.min(0),v.max(0));print('B targets',v)
for name,fun,init in [('tanh',np.tanh,1),('sin',np.sin,1),('softsign',lambda z:z/np.sqrt(1+z*z),1),('rational',lambda z:z/(1+z*z),.5)]:
 for linear in (False,True):
  for shifted in (False,True):
   def design(z):
    g=fun(u*np.exp(z[:4])+(z[4:] if shifted else 0));return aff(np.column_stack([u,g]) if linear else g)
   z0=np.r_[np.full(4,np.log(init)),np.zeros(4) if shifted else []]; z=least_squares(lambda z:regress(design(z),v)[1].ravel(),z0,max_nfev=150,bounds=(np.r_[[-3]*4,[-2]* (4 if shifted else 0)],np.r_[[2]*4,[2]* (4 if shifted else 0)])).x;d=design(z); w,r=regress(d,v); h=np.sum(np.linalg.pinv(d).T*d,axis=1)[:,None];print('B premix',name,linear,shifted,np.sqrt(np.mean(r*r)), np.sqrt(np.mean((r/(1-h))**2)),z)
 for fixed in (False,True):
  scores=[];params=[]
  for j in range(4):
   y=v[:,j];d=aff(u);w,_=regress(d,y)
   def pred(z):return (0 if fixed else z[5])+ (1 if fixed else np.exp(z[6]))*fun(d@z[:5]);
   sol=least_squares(lambda z:pred(z)-y,np.r_[w,[] if fixed else [0,0]],max_nfev=400);scores.append(np.sqrt(np.mean(sol.fun**2)));params.append(sol.x)
  print('B postmix',name,fixed,scores,np.array(params))
