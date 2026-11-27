exec(open('/tmp/m98_probe.py').read().split('for label,z in')[0])
At,Rt=m._design(taus,dt,t,train); ct=tr['strain_curves_obs']-tr['strain_0'][:,None]; rt=tr['instant_rate_obs']; e=At@theta-ct; er=Rt@theta-rt
print('train abs curve/rate',np.quantile(abs(e),[.5,.9,.99,1]),np.quantile(abs(er),[.5,.9,.99,1])); print('train curve median by time',np.median(e,axis=0));print('train lab median errors',[(i,np.median(e[tr['lab_id']==i]),np.median(er[tr['lab_id']==i])) for i in np.unique(tr['lab_id'])]);
for name,z in [('frozen',frozen(g)),('jointTD',m._aligned_features(tr,g)[-len(cy):]),('diagOnly',m._aligned_features(g,g)[:len(cy)])]:
 P,H,_=m._coupon(g,z,c); A,R=m._design(taus,dt,t,(P,H,z)); dr=z@theta[2:]; C=A@theta; rate=R@theta
 u=G*taus-t[:,None]; E=(cy-C)-dr[:,None]*(u@np.array([.62686,.60591])); S=ry-rate
 print(name,'lab/rate/early/mid/late/drift/hold');print(np.column_stack([g['lab_id'],S,E[:,0],np.median(E[:,2:5],axis=1),np.median(E[:,-3:],axis=1),dr,H]).round(5))
 print('adjust rate drift median err',np.median(abs(E-S[:,None]*(t+u@np.array([.62686,.60591])))))
print('asset index top keys'); import json; index=json.load(open('/holdout_task/public_asset_index.json'));print(list(index) if isinstance(index,dict) else type(index));
