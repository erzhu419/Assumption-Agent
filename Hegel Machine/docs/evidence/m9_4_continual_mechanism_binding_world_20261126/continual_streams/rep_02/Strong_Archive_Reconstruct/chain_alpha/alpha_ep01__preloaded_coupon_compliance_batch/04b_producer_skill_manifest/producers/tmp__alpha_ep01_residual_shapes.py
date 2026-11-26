import pickle, sys, numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_ep01_solver as s
d=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]['dataset']
a,b,y,c=s._prepare(d); c=c[:len(y)]; tr=d['train']; labs=tr['lab_id']; st=tr['stratum']; t=d['eval_times']; r=np.empty_like(y)
for lab in np.unique(labs):
    tau,w=s._fit(a,y,(labs!=lab)&c)
    r[labs==lab]=(y-a.matrix(np.log(tau))@w)[labs==lab]
z=r[:,:-1]-.25*r[:,-1,None]*t
late=np.abs(z[:,-1]); idx=np.flatnonzero(c); idx=idx[np.argsort(late[idx])[-20:][::-1]]
for j in idx:
    print(j,'lab,stratum',labs[j],st[j],'hold',round(a.hold[j],3),'strain0',round(tr['strain_0'][j],3),'rate_resid',round(r[j,-1],4),'curve_resid',np.round(z[j],3))
for j in range(len(t)):
    print('time',round(t[j],3),'abs_resid_quantiles',np.round(np.quantile(np.abs(z[c,j]),[.5,.9,.94,.95,.97,1]),4))
