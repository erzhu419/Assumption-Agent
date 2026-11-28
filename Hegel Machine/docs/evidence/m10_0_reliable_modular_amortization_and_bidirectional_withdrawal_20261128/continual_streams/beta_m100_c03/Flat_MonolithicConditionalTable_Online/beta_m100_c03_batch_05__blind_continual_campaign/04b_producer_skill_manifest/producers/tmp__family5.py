import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; q=c['single_bank_sweeps']; clean=np.max(abs(y),axis=1)<5
families={
 'rational':(lambda x,p:x/(1+p[0]*x*x),[.5],[0],[20]),
 'rational+tail':(lambda x,p:x/(1+p[0]*x*x)+p[1]*x,[.7,.2],[0,-10],[20,10]),
 'rationalpower':(lambda x,p:x/(1+p[0]*x*x)**p[1],[.5,1],[0,0],[20,20]),
 'rational-quartic':(lambda x,p:x/(1+p[0]*x*x+p[1]*x**4),[.4,0],[0,-.1],[20,20]),
 'rational-Pade':(lambda x,p:x*(1+p[1]*x*x)/(1+p[0]*x*x+p[2]*x**4),[.5,.1,0],[0,-5,-.1],[20,20,20]),
 'tanh-cubic':(lambda x,p:np.tanh(p[0]*x+p[1]*x**3),[1,.1],[.01,-2],[10,10]),
 'tanh-rational':(lambda x,p:np.tanh(p[0]*x)/(1+p[1]*x*x),[1,.1],[.01,-.1],[10,20]),
 'tanh+tail':(lambda x,p:np.tanh(p[0]*x)+p[1]*x,[1,0],[.01,-5],[10,5]),
 'erf':(lambda x,p:erf(p[0]*x),[1],[.01],[10]),
 'erf+tail':(lambda x,p:erf(p[0]*x)+p[1]*x,[1,0],[.01,-5],[10,5]),
 'atan-cubic':(lambda x,p:np.arctan(p[0]*x+p[1]*x**3),[1,.1],[.01,-2],[10,10]),
 'sin+tail':(lambda x,p:np.sin(p[0]*x)+p[1]*x,[1,0],[.01,-5],[3,5]),
 'poly7':(lambda x,p:x+p[0]*x**3+p[1]*x**5+p[2]*x**7,[-.3,.07,-.01],[-10]*3,[10]*3)
}
if __name__=='__main__':
 for name,(fun,p,lo,hi) in families.items():
  v=[]
  for j in range(4):
   m=(q['bank_id']==j)&(abs(q['raw_readout'])<5); a=q['ref_signal'][m]; b=q['raw_readout'][m]
   r=least_squares(lambda p:p[0]+p[1]*fun(a,p[2:])-b,[0,1,*p],bounds=([-1,.01,*lo],[1,10,*hi]),max_nfev=2000,gtol=1e-13);v.append((np.sqrt(np.mean(r.fun**2)),r.x.round(5).tolist()))
  print(name,v)
 print('archive wall definitions')
