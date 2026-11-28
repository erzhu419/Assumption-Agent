exec(open('/tmp/audit_batch.py').read().split('for name,a,b in folds:')[0])
terms=[k for degree in range(4) for k in combinations_with_replacement(range(4),degree)]
cubic=[k for k in terms if len(k)==3 and any(set(pair).issubset(k) for pair in support)]
u=forward_b(inverse_a(x)); ut=forward_b(inverse_a(s['test']['x']))
def design(u,terms): return np.column_stack([u[:,k].prod(1) for k in terms])
base=[k for k in terms if len(k)<2]; core=design(u,base+support+cubic); wide=design(u,[k for k in terms if len(k)<3]+cubic)
# Hold out rows nearest the actual test covariate extremes, using inputs only.
farthest=np.argsort(np.max(abs((u-u.mean(0))/u.std(0)),axis=1)); folds.append(('latent_axes',farthest[:52],farthest[52:]))
for name,a,b in folds:
 print(name, 'raw',metric(pred(f,a,b),b).round(5),'wide',metric(pred(wide,a,b),b).round(5),'compact',metric(pred(core,a,b),b).round(5),'compact-exp',metric(pred(core,a,b,'exp'),b).round(5))
for label,g,gt in [('compact',core,design(ut,base+support+cubic)),('wide',wide,design(ut,[k for k in terms if len(k)<3]+cubic)),('raw',f,m._design(s['test']['x']))]:
 vh=np.linalg.norm(gt@np.linalg.pinv(g),axis=1); print(label,'test leverage median, max',np.median(vh),max(vh))
p=s['panel_c']; d=m._design(p['u'])[:,:15]
for k in ('v1','v2'):
 co=ls(d,p[k]); print(k,'quadratic RMS',np.sqrt(np.mean((d@co-p[k])**2)), 'quadratic term norms',np.linalg.norm(co[5:],axis=1).round(5))
