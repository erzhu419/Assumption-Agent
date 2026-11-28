import pickle,numpy as np
from scipy.optimize import least_squares
from ros_probe import x,y,fit
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; bank=st['sensor_calibration_sweeps']['single_bank_sweeps']; z=bank['ref_signal']; sy=bank['raw_readout']; bi=bank['bank_id']; valid=fit(x,y,'affine')[1];
for j in range(4):
 q=(bi==j)&(np.abs(sy)<4); zz,yy=z[q],sy[q];
 for shift in (False,True):
  p=least_squares(lambda p:p[0]+p[1]*zz+p[2]*np.tanh(p[3]*zz+(p[4] if shift else 0))-yy,[0,.2,1,1.4,0] if shift else [0,.2,1,1.4]).x
  print('single',j,shift,p.round(6),np.sqrt(np.mean((p[0]+p[1]*zz+p[2]*np.tanh(p[3]*zz+(p[4] if shift else 0))-yy)**2)))
print('pulse structure',[(list(p) if isinstance(p,dict) else type(p).__name__) for p in (st.get('reference_tracer_pulses') or [])]); print('ref rows',x[:16].round(3));
# Separate own-channel and opposite-channel response, jointly fitting single-bank observations.
def design(x,k,j,mode):
 o=j^2; a=x[:,j]; b=x[:,o]; u=np.tanh(k[0]*a); v=np.tanh(k[1]*b); opts={'quad':b*b,'rad':b*b/(1+k[2]*b*b),'cos':1-np.cos(k[2]*b),'sech':v*v,'exp':-np.expm1(-k[2]*b*b)}; d=[np.ones(len(x)),a,u,b,v,opts[mode]]
 if k.size==4: d.append(b/(1+k[3]*b*b))
 return np.column_stack(d)
for mode in ('quad','rad','cos','sech','exp'):
 for extra in (False,True):
  err=[]; ps=[]
  for ids in [np.array([],int),*np.array_split(np.argsort(np.linalg.norm(x,axis=1)),5)]:
   train=valid.copy();train[ids]=False; row=[]; pf=[]
   for j in range(4):
    q=(bi==j)&(np.abs(sy)<4); xs=np.zeros((q.sum(),4));xs[:,j]=z[q]; xx=np.vstack((x[train],xs)); yy=np.r_[y[train,j],sy[q]]
    def project(k):
     d=design(xx,k,j,mode); return d,np.linalg.lstsq(d,yy,rcond=None)[0]
    def fun(k):
     d,p=project(k);return d@p-yy
    k=least_squares(fun,[1.5,1.5,1.5]+([1.] if extra else []),bounds=(.15,5),max_nfev=150).x; _,p=project(k);pf.append(np.r_[k,p]); v=(ids[valid[ids]] if ids.size else np.flatnonzero(valid)); row.append(y[v,j]-design(x[v],k,j,mode)@p)
   err.append(np.sqrt(np.mean(np.array(row)**2)));ps.append(pf)
  print(mode,extra,np.round(err,6)); print(np.array(ps[0]).round(4))
