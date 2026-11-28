exec(open('/tmp/probe.py').read().split('u,v=[s')[0]);z=np.load('/tmp/probe.npz')['z'];
def op(p,k,f,base=aff):
 u,v=s[p]['u'],s[p][k];D=lambda a,w:np.c_[base(a),f(w*a)]; grid=np.linspace(.15,5,100); j=np.argmin([report(D(u,w),v)[0] for w in grid]); w=minimize_scalar(lambda w:report(D(u,w),v)[0],bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded').x;c=fit(D(u,w),v);return lambda a:D(a,w)@c
id=lambda a:a
p=op('panel_c','v1',np.tanh);q=op('panel_c','v2',lambda a:a*a/(1+a*a)); u,v=s['panel_b']['u'],s['panel_b']['v'];b=lambda a:poly(a)@fit(poly(u),v); bb=op('panel_b','v',lambda a:a*a/(1+a*a));
def score(M):
 r=M[52:]@fit(M[:52],Y[:52])-Y[52:];return np.array([abs(r[:,0]).mean(),abs(r[:,1:]).mean()])
rs=[]
for nam,f in [('z',id),('p',p),('q',q),('b',b),('bb',bb),('qp',lambda x:q(p(x))),('pq',lambda x:p(q(x))),('bp',lambda x:b(p(x))),('pb',lambda x:p(b(x))),('bq',lambda x:b(q(x))),('qb',lambda x:q(b(x)))]:
 a=f(z); other=[z,p(z),q(z),b(z)];
 for name,M in [('aff',aff(a)),('quad',poly(a))]+[(str(i)+':'+str(j),np.c_[aff(a),o,a*np.roll(o,j,axis=1)]) for i,o in enumerate(other) for j in range(4)]:rs.append((score(M),nam,name))
for j in range(2):print('best',j,sorted(rs,key=lambda r:r[0][j])[:10]);
