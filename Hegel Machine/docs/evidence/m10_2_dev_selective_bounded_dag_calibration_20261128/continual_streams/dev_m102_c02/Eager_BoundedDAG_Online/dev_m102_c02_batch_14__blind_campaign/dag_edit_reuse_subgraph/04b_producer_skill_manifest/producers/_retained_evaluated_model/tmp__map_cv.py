import sys, pickle, numpy as np, importlib.util
sys.path.insert(0,'/holdout_task'); import node_00 as n
sys.path.insert(0,'/holdout_task/_frozen_nodes'); import node_01 as a,node_02 as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print(s.keys()); print({k:list(s[k]) for k in ('panel_a','panel_b','panel_c')}); m=n._fit_sensor(s['panel_a']); print('models',m); x=s['train']['x']; z=n._undo_sensor(x,m); print('ranges',np.min(z,0),np.max(z,0)); u=s['panel_a']['u']; v=s['panel_a']['v']; ok=np.max(abs(u-v),axis=1)<2; print('inverse panel max',np.max(abs(u[ok]-n._undo_sensor(v[ok],m)),0)); tr=s['train']['traj']; y=s['train']['y']; print('traj singular',np.linalg.svd(tr,compute_uv=False)); print('eval times',s['eval_times']);

def cv(name,c):
    vals=[]
    for test in [np.arange(52,72),np.arange(20),np.arange(26,46),np.arange(0,72,4)]:
        fit=np.setdiff1d(np.arange(72),test); st={'train':{'y':y[fit],'traj':tr[fit]},'eval_times':s['eval_times']}; o=a.run_node({'__input__':st,'node_00':{'c':c[fit],'query':c[test]}}); p=b.run_node({'node_01':o})['state_pred']; vals.append([np.mean(abs(o['y_hat']-y[test])),np.mean(abs(p-tr[test]))]);
    print(name,np.round(vals,5).tolist())
cv('raw',x); cv('z',z)
for pn,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 for kind in ['linear','quad','cube','sin','tanh','atan','rat']:
    cv((pn,k,kind),n._map_coordinates(s[pn],z,kind,k))
for p,q in [('panel_b','panel_c'),('panel_c','panel_b')]:
 for k in ['v1','v2']:
    F=lambda zz:n._map_coordinates(s['panel_b'],zz,'sin','v'); G=lambda zz:n._map_coordinates(s['panel_c'],zz,'atan' if k=='v1' else 'quad',k)
    cv(p+' then '+q+k,G(F(z)) if p=='panel_b' else F(G(z)))
print('panel relation',[(p,np.max(abs(s[p]['u']),0)) for p in ['panel_b','panel_c']])
