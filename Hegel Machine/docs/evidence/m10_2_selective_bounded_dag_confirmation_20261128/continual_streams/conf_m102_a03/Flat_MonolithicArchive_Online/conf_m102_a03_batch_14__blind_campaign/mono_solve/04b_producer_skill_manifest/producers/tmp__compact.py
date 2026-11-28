exec(open('/tmp/graph_probe.py').read().split("B=panel(")[0]); np.set_printoptions(precision=5,suppress=True,linewidth=140)
u=s['panel_b']['u']; v=s['panel_b']['v'];
def lin(A,Y):return np.linalg.lstsq(A,Y,rcond=1e-10)[0]
models=[]
def probe(name,f,starts,lo,hi):
 def err(p):
  A=af(f(u,p)); return (A@lin(A,v)-v).ravel()
 q=min((least_squares(err,p,bounds=(lo,hi),max_nfev=120) for p in starts),key=lambda q:q.cost); A=af(f(u,q.x)); c=lin(A,v); models.append((np.std(q.fun),name,q.x,c,lambda x:f(x,q.x)))
for k,f in F.items():
 for l,g in {**F,'cube':lambda x,p:x**3,'signedsq':lambda x,p:x*abs(x)}.items():
  if k>=l and l in F: continue
  probe(k+' '+l,lambda x,p,f=f,g=g:np.c_[x,f(x,p[0]),g(x,p[1])],[[1.5,2.8],[2.8,1.5],[.8,.8]], [.01,.01],[7,7])
for k,f in F.items():
 probe(k+' mixed',lambda x,p,f=f:np.c_[x,f(x,p[0]),x*np.roll(x,1,axis=1),x*np.roll(x,2,axis=1)[:,:] if False else x[:,0:2]*x[:,2:4]], [[1.5],[2.8],[.8]],[.01],[7]);
 probe(k+' cube2',lambda x,p,f=f:np.c_[x,f(x,p[0]),x**2],[[1.5],[2.8],[.8]],[.01],[7]);
 probe(k+' sinusoidal channel',lambda x,p,f=f:np.c_[x,f(x,p)],[[1.5]*4,[2.8]*4,[.8]*4],[.01]*4,[7]*4)
print('panel ranking',[(a,b,c) for a,b,c,_,_ in sorted(models,key=lambda m:m[0])[:23]])
for a,b,p,c,f in sorted(models,key=lambda m:m[0])[:5]:print(b,c)
S=np.load('/tmp/spaces.npz'); viz=[]
for name,b in [(k,S[k]) for k in ['B0','B2','B3']]+[(m[1],af(m[4](z))@m[3]) for m in sorted(models,key=lambda m:m[0])[:10]]:
 pair=z*np.roll(b,1,axis=1); P=pair.sum(1); Y=np.c_[y,T];
 for j,D in [('P',np.c_[af(b),P]),('zP',np.c_[af(b),z,P]),('pair',np.c_[af(b),pair]),('zpair',np.c_[af(b),z,pair]),('Psq',np.c_[af(b),P,P**2]),('linearzPsq',np.c_[af(b),z,P,P**2])]:
  c=lin(D[:52],Y[:52]); e=D[:72]@c-Y; viz.append((np.mean(abs(e[52:,0])),np.mean(abs(e[52:,1:])),np.mean(abs(e[:52,0])),name,j));
 if name=='B0':
  D=np.c_[af(b),z,P]; c=lin(D[:52],Y[:52]); print('coef simple',c[:,[0,1,2,8]]); print('resid sorted',np.c_[z[:72],b[:72],P[:72],Y[:,0],Y[:,0]-D[:72]@c[:,0]][np.argsort(np.abs(Y[:,0]-D[:72]@c[:,0]))[-12:]])
print('readouts',sorted(viz)[:24]);
