import pickle,numpy as np
from scipy.optimize import least_squares
q=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=q['calibration_load_cells']; u=c['true_channel_loads']; r=c['bridge_raw']; s=c['single_channel_sweeps']; np.set_printoptions(precision=5,suppress=True)
ps=[]
for j in range(4):
 m=(s['lab_id']==j)&(abs(s['cmd_stress'])<3); a,y=s['true_stress'][m],s['cmd_stress'][m]; ps.append(least_squares(lambda p:p[0]+p[1]*a+p[2]*np.sin(p[3]*a)-y,[0,1,.25,2.5]).x)
p=np.array(ps).T
v=r-p[0]-p[1]*u-p[2]*np.sin(p[3]*u)
w=u[:,[2,3,0,1]]
for j in range(4):
 keep=abs(v[:,j])<2; a=u[:,j]; b=w[:,j]; y=v[:,j]; print('channel',j,'axis coupling',np.c_[b[(a==0)&keep&(b!=0)],y[(a==0)&keep&(b!=0)]])
 for name,df in [('poly',lambda x:[x,x**2,x**3,x**4,x**5,x**6]),('trig',lambda x:[x,np.sin(2.5*x),np.cos(2.5*x)-1]),('trig2',lambda x:[x,np.sin(1.25*x),np.cos(1.25*x)-1,np.sin(2.5*x),np.cos(2.5*x)-1]),('tanh',lambda x:[x,np.tanh(x),np.tanh(x)**2]),('rat',lambda x:[x,x/(1+x*x),x*x/(1+x*x)]),('full',lambda x:[x,x**2,np.sin(2.5*x),np.cos(2.5*x)-1,np.tanh(x),np.tanh(2.5*x)])]:
  A=np.column_stack(df(b)+[a*b]); coef=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0]; e=A@coef-y; folds=np.arange(len(a))%5; cv=np.zeros(len(a));
  for k in range(5):
   fit=keep&(folds!=k); cv[folds==k]=A[folds==k]@np.linalg.lstsq(A[fit],y[fit],rcond=None)[0]-y[folds==k]
  print(name,np.quantile(abs(cv[keep]),[.5,.9,1]),coef)
 for label, fun in [('mixed input',lambda p:p[0]*(a+p[1]*b)+p[2]*np.sin(p[3]*(a+p[1]*b))+p[4]*(np.cos(p[5]*b)-1)+p[6]*a*b),('frequency',lambda p:p[0]*b+p[1]*np.sin(p[2]*b)+p[3]*(np.cos(p[4]*b)-1)+p[5]*a*b),('rational',lambda p:p[0]*b+(p[1]*b+p[2]*b*b)/(1+p[3]**2*b*b)+p[4]*a*b),('tanh shift',lambda p:p[0]*b+p[1]*(np.tanh(p[2]*b+p[3])-np.tanh(p[3]))+p[4]*a*b)]:
  y1=y+p[1,j]*a+p[2,j]*np.sin(p[3,j]*a) if label=='mixed input' else y
  start={'mixed input':[p[1,j],.2,p[2,j],p[3,j],.1,1.5,.04],'frequency':[.3,.02,2,.1,1.5,.04],'rational':[.3,.02,-.06,.3,.04],'tanh shift':[.3,.1,1,.3,.04]}[label]
  f=least_squares(lambda k:(fun(k)-y1)[keep],start,max_nfev=500); print(label,np.quantile(abs(f.fun),[.5,.9,1]),f.x)
