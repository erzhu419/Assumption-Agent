exec(open('/tmp/wall_probe.py').read().split("print('baseline by pulse'")[0])
def matrix(rows, cols, v):
 return coo_matrix((np.broadcast_to(v,rows.shape),(rows,cols)),shape=(N,N)).tocsc()
G=[]; H=[]; W=[]; F=[]; V=[]; R=[]; D=[]
for j,(rows,off) in enumerate(walls):
 a='xy'[j//2]; h=s['L'+a]/(s['n'+a]-1); k=s['kappa_'+a]; v=s['vel_'+a]; end=0 if j%2==0 else -1; step=1 if j%2==0 else -1; inner=end+step; span=s['n'+a]-1
 G.append(M[2*j+1]); H.append(matrix(rows,rows+2*off,1)-matrix(rows,rows+off,1)); W.append(matrix(rows,rows,1)-matrix(rows,rows+span*off,1)); F.append((k[inner]-k[end])/(2*h*h)*G[-1]); V.append(step*v[end]/h*G[-1]); R.append(s['robin_'+a]/h*M[2*j]); D.append(M[2*j])
models={'nearest+second':G+H,'nearest+wrap':G+W,'wrap':W,'wrap symmetric':[W[0]+W[1],W[2]+W[3]],'second symmetric':[H[0]+H[1],H[2]+H[3]],'face+upwind+leak':[sum(F),sum(V),sum(R)],'face+upwind+wrap':[sum(F),sum(V),W[0]+W[1],W[2]+W[3]],'nearest+second+wrap':G+H+W,'diag+nearest+second':D+G+H,'diag+nearest+wrap':D+G+W}
for name,Ms in models.items():
 scores=[]
 for hold in [0,1,2,-1]:
  keep=np.arange(3)!=hold
  fit=least_squares(lambda t:pr(pred(op(t,Ms))-y)[:,keep].ravel(),np.zeros(len(Ms)),max_nfev=120,gtol=1e-11)
  err=np.sqrt(np.mean(pr(pred(op(fit.x,Ms))-y)**2,axis=0)); scores.append(err[hold] if hold>=0 else max(err))
 print(name,np.round(scores,8),np.round(fit.x,5))
print('baseline by pulse',np.sqrt(np.mean(pr(pred(L)-y)**2,axis=0)))
