import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
q=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=q['dataset'];c=d['calibration_load_cells'];u=c['true_channel_loads'];r=c['bridge_raw'];s=c['single_channel_sweeps'];np.set_printoptions(precision=5,suppress=True,linewidth=180)
for j in range(4):
 m=s['lab_id']==j; a=np.r_[s['true_stress'][m],u[4*j:4*j+4,j]]; y=np.r_[s['cmd_stress'][m],r[4*j:4*j+4,j]]; sel=np.abs(y)<3; a,y=a[sel],y[sel]; print('axis',j)
 for name,f in [('tanh',np.tanh),('erf',erf),('atan',np.arctan),('rational',lambda z:z/np.sqrt(1+z*z)),('sin',np.sin)]:
  best=min((least_squares(lambda p:p[0]+p[1]*a+p[2]*f(a*p[3])-y,[0.1,.3,1.,k],bounds=([-2,-2,-4,.05],[2,4,4,8]),max_nfev=800) for k in [.6,1.4,2.7]),key=lambda z:z.cost); print(name,best.x,np.sqrt(2*best.cost/len(y)))
print('paired calibration excluding first 16 rows'); print(np.c_[u,r][16:]); print('dataset keys',list(d))
for k in ['train','test']:
 v=d[k]['aux_chamber_telemetry'];print(k, 'aux means',v.mean(0),'std',v.std(0))
