exec(open('/tmp/compare_walls.py').read().split('for a,b in itertools.product')[0])
bank=np.asarray(s['row_bank_id']); shift=np.repeat(np.arange(len(mus)),len(x)); xx=np.tile(x,len(mus)); yy=np.tile(yidx,len(mus)); xwall=(xx==0)|(xx==nx-1); ywall=(yy==0)|(yy==ny-1)
# Test symmetric seam withdrawal independently of removal of the endpoint upwind flux.
def op(d,kind):
 if kind!='S': return axis(d,kind)
 a=axis(d,'P').tolil(); k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
 for i,j,adv in [(0,-1,max(v[0],0)),(-1,0,max(-v[-1],0))]:
  q=(k[i]+k[j])/(2*h*h)+adv/h; a[i,i]-=q; a[i,j]+=q
 return a.tocsc()
for a,b in ('PP','PR','PS','SP','SS'):
 l=kron(eye(ny),op('x',a))+kron(op('y',b),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')); A=np.concatenate([splu((l+m*eye(nx*ny)).tocsc()).solve(rhs)[x+nx*yidx] for m in mus]); print(a+b,end=' ')
 for g in (bank,shift):
  errs=[]
  for i in np.unique(g):
   test=g==i; w,gamma=fit(A,~test); e=A[test]@w+z[test]@gamma-y[test]; errs.append(np.sqrt(np.mean(e*e)))
  print(np.round(errs,6),end=' ')
 w,gamma=fit(A,np.ones(len(y),bool)); e=A@w+z@gamma-y; print('bank biases',[round(e[bank==i].mean(),6) for i in np.unique(bank)],'wall RMS',np.round([np.sqrt(np.mean(e[j]**2)) for j in (xwall,ywall,~(xwall|ywall))],6))
