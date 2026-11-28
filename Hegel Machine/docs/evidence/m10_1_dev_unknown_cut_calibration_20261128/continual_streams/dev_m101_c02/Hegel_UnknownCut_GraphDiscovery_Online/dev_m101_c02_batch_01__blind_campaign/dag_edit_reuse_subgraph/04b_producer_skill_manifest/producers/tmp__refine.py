exec(open('/tmp/shared_basis.py').read().split('models={}')[0])
from scipy.optimize import least_squares

def mat(u,kind,w):
    cols=[np.ones(len(u)),u]; k=0
    for t in kind:
        if t=='q': cols.append(u*u); continue
        a=w[k:k+4]*u; k+=4
        cols.append({'h':np.tanh,'s':np.sin,'c':np.cos}[t](a))
    return np.column_stack(cols)

def fit(u,y,kind,start):
    def fun(w):
        A=mat(u,kind,w); return (A@np.linalg.lstsq(A,y,rcond=None)[0]-y).ravel()
    w=least_squares(fun,np.repeat(start,4),bounds=(.35,6),max_nfev=180).x
    A=mat(u,kind,w); return w,np.linalg.lstsq(A,y,rcond=None)[0],np.sqrt(np.mean(fun(w)**2)),loo(A,y).mean()

a=s['panel_a'];good=np.max(abs(a['v']-a['u']),axis=1)<1;u=a['u'][good];v=a['v'][good];g=((u-u.mean(0))*(v-v.mean(0))).sum(0)/((u-u.mean(0))**2).sum(0);o=v.mean(0)-g*u.mean(0);x=(s['train']['x']-o)/g;Y=np.column_stack([s['train']['y'],s['train']['traj']]); maps={}
for p,v,configs in [('panel_b','v',[('h',[1.35]),('s',[2.8]),('qh',[1.35]),('hs',[1.35,2.8])]),('panel_c','v1',[('h',[2.2]),('qh',[2.2]),('hs',[2.2,3.35])]),('panel_c','v2',[('qc',[2.48]),('c',[1.27])])]:
    for t,w0 in configs:
        w,B,rmse,l=fit(s[p]['u'],s[p][v],t,w0);print(v,t,'w',w,'RMSE',round(rmse,7),'fixed-map LOO',round(l,7));
        if t in ['h','c','qc']: print('coefficients',B)
        maps[v+t]=mat(x,t,w)@B

def assess(A):
    z=loo(A,Y); b=np.linalg.lstsq(A[:52],Y[:52],rcond=None)[0]; e=np.abs(A[52:]@b-Y[52:]).mean(0);return np.round([z[0],z[1:].mean(),z[-1],e[0],e[1:].mean(),e[-1]],5)
for k in ['vh','vs','vqh','vhs']:
    b=maps[k];c=maps['v1h'];d=maps['v2qc'];
    for label,cols in [('b',[b]),('bc',[b,c]),('bcd',[b,c,d]),('bc+bb',[b,c,basis(b,'qp')[:,5:]]),('bc+bd',[b,c,b*d]),('bc+c2',[b,c,c*c]),('bc+tanhB',[b,c,np.tanh(b)]),('bc+tanhC',[b,c,np.tanh(c)]),('bc+Bsin',[b,c,np.sin(2*b)])]:
        print(k,label,assess(np.column_stack([np.ones(len(x)),*cols])))
