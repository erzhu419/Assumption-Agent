import runpy,pickle,numpy as np,time
m=runpy.run_path('/holdout_task/node_00.py'); f=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node'];s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; z=m['_align'](s['panel_a'],x); print('z',z.min(0),z.max(0)); p=s['panel_a']; j=np.max(abs(p['v']),axis=1)<3.5; print('align',np.quantile(abs(m['_align'](p,p['v'][j])-p['u'][j]),[.5,.95,1])); c=s['panel_c']; q,r=[m['_branch'](c['u'],c['v'+str(k)],z,k) for k in (1,2)]; print('branches',[(a[0],a[2:]) for a in (q,r)]);q,r=q[1],r[1];p=m['_pair'](z); l=m['_ones'](z)@m['_ls'](m['_ones'](s['panel_b']['u']),s['panel_b']['v']); Y=np.c_[t['y'],t['traj']];
F={'p':p,'q':q,'r':r,'z':z,'l':l,'c24':np.cos(2.4*z),'t15':np.tanh(1.5*z),'s24':np.sin(2.4*z),'rat':z/(1+.85*z*z),'soft':z/(1+.85*abs(z))};out=[]
for a in F:
 for b in F:
  if a>=b:continue
  A,B=F[a],F[b];D=m['_ones'](np.c_[A,B,(A[:,:,None]*B[:,None,:]).reshape(len(A),-1)]);P=np.linalg.pinv(D);E=Y-D@(P@Y);H=(D*P.T).sum(1); er=np.mean(abs(E/(1-H[:,None])),axis=0);out.append((er[0]+er[1:].mean(),a,b,er[0],er[1:].mean(),np.linalg.svd(E,compute_uv=False)[:3]))
print('linear loo',*sorted(out)[:15],sep='\n');
for a,b in [('r','q'),('p','c24'),('p','q'),('p','r'),('q','z'),('r','z')]+[(a,b) for _,a,b,*_ in sorted(out)[:5]]:
 es=[]
 for seed in (44,91,153):
  perm=np.random.default_rng(seed).permutation(len(x)); i,j=perm[:54],perm[54:];k=perm; ss={'train':{h:t[h][i] for h in ('x','y','traj')}}; O=f({'__input__':ss,'node_00':dict(z=z[k],L=l[k],Q=q[k],R=r[k],product=F[a][k],cosine=F[b][k])});es.append((np.mean(abs(O['y_hat']-t['y'][j])),np.mean(abs(O['state_pred']-t['traj'][j]))))
 print(a,b,np.round(es,5),flush=True)
