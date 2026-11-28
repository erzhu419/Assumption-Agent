exec(open('/tmp/probe.py').read().split('u,v=')[0])
u,v=(s['panel_a'][k] for k in ['u','v']); X=one(u); q=ls(X,v); r=np.linalg.norm(X@q-v,axis=1); good=r<4*np.median(r)
for j in range(4):
 a=u[:,j]; b=u[:,(j+1)%4]; d=one(np.c_[a,b,a*b,a**3]); p=ls(d[good],v[good,j]); print('adjacent',j,p,np.quantile(abs(d[good]@p-v[good,j]),[.5,1]))
 D=one(np.column_stack([np.prod(u[:,idx],axis=1) for k in range(1,4) for idx in combinations_with_replacement(range(4),k)])); p=ls(D[good],v[good,j]); print('cubic err',np.std(D[good]@p-v[good,j])); print('dominant',sorted(zip([()]+[idx for k in range(1,4) for idx in combinations_with_replacement(range(4),k)],p),key=lambda a:-abs(a[1]))[:12])
print('panel c ranges',[(k,v.min(0),v.max(0)) for k,v in s['panel_c'].items()]); print('B matrix',ls(one(s['panel_b']['u']),s['panel_b']['v']))
