exec(open('/tmp/compose.py').read().split('features={};')[0])
print('B coefficients',fit); print('C1 coefficients',m._ols(m._products(u),p['v1'])); print('B/C correlations'); print({k:{j:np.shape(v) for j,v in s[k].items()} for k in ['panel_a','panel_b','panel_c']})
# Examine whether calibrated latent coordinates or a composed map linearizes the target.
from itertools import combinations_with_replacement as cr
base={**coords,'P':prod(z),'C':cos(z),'P+C':prod(z)+cos(z),'P*C':prod(z)*cos(z),'P(C)':prod(cos(z)),'C(P)':cos(prod(z)),'B(P)':bp(prod(z)),'B(C)':bp(cos(z)),'P(Bc)':prod(coords['Bc']),'C(Bc)':cos(coords['Bc'])}
D={}
for k,a in base.items():
 for name,val in [('id',a),('prod',prod(a)),('cos',cos(a)),('sin',np.sin(q*a)),('soft',a/(1+abs(a))),('mix',prod(a)+cos(a))]: D[k+'/'+name]=val
for k in ['Z','B','I','Bc','Ic','X']:
 a=base[k]; D[k+'/allprod']=np.column_stack([a[:,i]*a[:,j] for i,j in cr(range(4),2)]); D[k+'/triple']=np.column_stack([np.prod(a[:,list(t)],axis=1) for t in I.combinations(range(4),3)]); D[k+'/cospair']=np.column_stack([np.cos(q*(a[:,i]+sign*a[:,j])) for i,j in I.combinations(range(4),2) for sign in [1,-1]]); D[k+'/sinpair']=np.column_stack([np.sin(q*(a[:,i]+sign*a[:,j])) for i,j in I.combinations(range(4),2) for sign in [1,-1]])
rank=[]
for k,a in D.items():
 for h,b in [('',None),*D.items()]:
  if h and h<=k: continue
  A=a if b is None else np.c_[a,b]; r=score(A); rank.append((np.max(r),np.max(r[0]),k,h,r))
for i in [0,1]:
 print('ranking',i)
 for row in sorted(rank,key=lambda r:r[i])[:12]: print(*row[:4],row[4].round(4).tolist())
# Sparse coefficient recovery on elementary harmonics and coordinate monomials.
A=[]; names=[]
for k,a in D.items():
 if k.split('/')[0] not in ['Z','P','C','B','I','Bc','Ic']: continue
 for j,col in enumerate(a.T): A.append(col); names.append(k+':'+str(j))
A=np.stack(A,1); A=(A-A.mean(0))/np.maximum(A.std(0),1e-8); Y=y[:,[0,-1]]
for j in range(2):
 selected=[]; res=Y[:,j]-Y[:,j].mean(); Q=np.c_[np.ones(n),A]
 for i in range(16):
  idx=np.argmax(abs(A.T@res)); selected.append(idx); B=Q[:,[0,*[a+1 for a in selected]]]; co=np.linalg.lstsq(B,Y[:,j],rcond=None)[0]; res=Y[:,j]-B@co
  if i<9: print('OMP',j,i,names[idx],np.linalg.norm(res)/np.sqrt(n))
