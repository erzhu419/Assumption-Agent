exec(open('/tmp/chamber_probe.py').read().split('a=cal')[0])
np.set_printoptions(precision=4,suppress=True,linewidth=148)
a=cal['true_channel_loads']; b=cal['bridge_raw'];
print('cal rows load | raw | sums',np.c_[a,b,a.sum(1),b.sum(1)][:25]); print('sweeps:',np.column_stack(list(cal['single_channel_sweeps'].values()))); print('range cal train diag',[(z.min(0),z.max(0)) for z in [b,tr['past_channel_raw'].reshape(-1,4),g['past_channel_raw'].reshape(-1,4)]]);
print('reference groups by lab: mean/std across all panel columns, feature mean:')
for name,z in [('tr',tr),('te',te),('g',g)]:
 for k in range(4):
  rows=z['lab_id']==k; p=z['ref_coupons'][rows];print(name,k,np.median(p.mean(0)),np.median(p.std(0)),np.nanmean(z['coupon_features_raw'][rows],0))
for label,x,y in [('forward',a,b),('inverse',b,a)]:
 def basis(x,degree):
  from itertools import combinations_with_replacement
  cols=[np.ones(len(x))]; names=[()]
  for k in range(1,degree+1):
   for c in combinations_with_replacement(range(4),k):cols.append(np.prod(x[:,c],axis=1)); names.append(c)
  return np.array(cols).T,names
 for degree in (1,2,3):
  A,names=basis(x,degree); Q=np.linalg.pinv(A); w=Q@y; e=(A@w-y)/(1-np.sum(A*Q.T,axis=1))[:,None]; print(label,degree,'LOO channel RMSE',np.sqrt(np.mean(e**2,axis=0)), 'sum RMSE',np.sqrt(np.mean(e.sum(1)**2)),'largest axial coeffs',[(names[j],round(w[j].sum(),4)) for j in np.argsort(abs(w.sum(1)))[-8:]])
