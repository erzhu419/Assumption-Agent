import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; panel=s['panel_b']; u,v=panel['u'],panel['v']; errors=[]
for i in range(len(u)):
    k=np.arange(len(u))!=i
    pred=m._operator({'u':u[k],'v':v[k]},u[i:i+1])[0]
    f=m._quadratic(u); base=f[i]@np.linalg.lstsq(f[k],v[k],rcond=None)[0]
    errors.append([pred-v[i],base-v[i]])
rmse=np.sqrt(np.mean(np.square(errors),axis=(0,2)))
print('ng_t06 | exact saved candidate panel-B leave-one-out RMSE:',rmse[0],'| raw-quadratic ablation:',rmse[1]); assert rmse[0]<0.001 and rmse[0]<0.01*rmse[1]
