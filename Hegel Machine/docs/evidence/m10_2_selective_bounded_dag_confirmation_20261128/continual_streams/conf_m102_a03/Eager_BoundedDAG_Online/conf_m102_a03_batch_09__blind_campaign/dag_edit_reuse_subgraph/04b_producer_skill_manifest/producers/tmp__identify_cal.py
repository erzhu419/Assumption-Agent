import pickle,itertools,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=6,suppress=True,linewidth=150)
u,v=(s['panel_a'][k] for k in ('u','v')); a=np.c_[np.ones(len(u)),u]
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
err=np.linalg.norm(a@fit(a,v)-v,axis=1); good=err<4*np.median(err);print('excluded',np.where(~good)[0]);u,v=u[good],v[good]
terms=[t for d in range(4) for t in itertools.combinations_with_replacement(range(4),d)]
m=np.column_stack([np.prod(u[:,t],axis=1) for t in terms]);
for j in range(4):
 x,r,l=u[:,j],u[:,(j+1)%4],u[:,(j-1)%4];print('\nout',j)
 for tag,b in [('affine',m[:,:5]),('cyclic',np.c_[np.ones(len(x)),x,x**3,r,l]),('cycle_interaction',np.c_[np.ones(len(x)),x,x**3,r,l,x*r,x*l,x*x*r,x*r*r,x*x*l,x*l*l]),('all_cubic',m)]:
  c=fit(b,v[:,j]); e=b@c-v[:,j]; print(tag,'rms',np.std(e),'max',max(abs(e)),c if tag!='all_cubic' else sorted(zip(terms,c),key=lambda t:-abs(t[1]))[:13])
 # Sparse orthogonal matching pursuit, with optional forward selection by exact LS improvement.
 ix=[0,j+1];
 for z in range(9):
  if z: ix.append(min((k for k in range(1,len(terms)) if k not in ix),key=lambda k:np.linalg.norm(m[:,ix+[k]]@fit(m[:,ix+[k]],v[:,j])-v[:,j])))
  c=fit(m[:,ix],v[:,j]);print('sparse',[(terms[k],round(b,6)) for k,b in zip(ix,c)],'rms',round(np.std(m[:,ix]@c-v[:,j]),8))
