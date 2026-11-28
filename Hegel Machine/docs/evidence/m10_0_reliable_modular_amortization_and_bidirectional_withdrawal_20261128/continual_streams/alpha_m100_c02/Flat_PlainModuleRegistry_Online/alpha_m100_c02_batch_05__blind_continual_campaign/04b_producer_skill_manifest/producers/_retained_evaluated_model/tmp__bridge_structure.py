import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; keep=np.max(abs(y),axis=1)<4; x,y=x[keep],y[keep]; axis=np.count_nonzero(x,axis=1)==1
I,J=np.array(list(combinations_with_replacement(range(4),2))).T
partners=[2,3,0,1]
def fit(x,y,j,typ):
 p=partners[j]
 def design(x,s):
  u=x[:,j]; v=x[:,p]; z=u+s[1]*v
  a=[np.ones(len(x)),*x.T]
  if typ.startswith('single'): a += [np.tanh(s[0]*z+s[2]),np.cos(s[3]*v),np.sin(s[3]*v)]
  elif typ.startswith('separate'): a += [np.tanh(s[0]*u+s[2]),np.tanh(s[3]*v),np.cos(s[4]*v),np.sin(s[4]*v)]
  elif typ.startswith('double'): a += [np.tanh(s[0]*z+s[2]),np.tanh(s[3]*v),np.cos(s[4]*u),np.cos(s[4]*v)]
  elif typ.startswith('fourier'): a += [np.sin(s[0]*u),np.cos(s[0]*u),np.sin(s[3]*v),np.cos(s[3]*v)]
  if typ.endswith('q'): a.extend((x[:,I]*x[:,J]).T)
  if typ.endswith('pair'): a.extend([u*v,(u*v)**2,u*u*v,u*v*v])
  return np.column_stack(a)
 def profile(s,ret=False):
  a=design(x,s); b=np.linalg.lstsq(a,y,rcond=1e-10)[0]; return b if ret else a@b-y
 o=least_squares(profile,[1.5,0.,0.,1.5,2.],bounds=([.3,-.7,-1.,.3,.4],[4,.7,1.,5.,6.]),max_nfev=80)
 b=profile(o.x,True); return o.x,b,lambda t:design(t,o.x)@b
for typ in ['single','singlepair','singleq','separate','separatepair','separateq','double','doublepair','doubleq','fourier','fourierpair','fourierq']:
 r=np.empty_like(y); pars=[]; bs=[]; tr=[]
 for j in range(4):
  s,b,f=fit(x,y[:,j],j,typ); pars.append(s); bs.append(b); tr.append(np.sqrt(np.mean((f(x)-y[:,j])**2)))
  for ii in np.array_split(np.random.default_rng(23).permutation(len(x)),8):
   m=np.ones(len(x),bool);m[ii]=False; _,_,f=fit(x[m],y[m,j],j,typ);r[ii,j]=f(x[ii])-y[ii,j]
 print(typ,'fit',np.round(tr,6),'OOF',np.round(np.sqrt(np.mean(r*r,axis=0)),6),'axis/mixed maxima',np.round([abs(r[axis]).max(),abs(r[~axis]).max()],6))
 if min(tr)<.004: print('params',np.round(pars,4),'coef',np.round(bs,4))
