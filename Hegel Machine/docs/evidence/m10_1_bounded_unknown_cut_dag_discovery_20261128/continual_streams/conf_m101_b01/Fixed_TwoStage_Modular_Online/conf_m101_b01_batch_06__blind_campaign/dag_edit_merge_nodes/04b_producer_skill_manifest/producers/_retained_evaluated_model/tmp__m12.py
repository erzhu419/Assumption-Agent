import pathlib,pickle,sys
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve
sys.path.insert(0,'/holdout_task')
import node_00 as old
p=pathlib.Path('/holdout_task/node_00.py'); source=p.read_text(); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
exec('''
def _new_b(panel, x, mode):
    u,v=panel['u'],panel['v']
    def phi(x,t):
        a,b=t
        if mode==0: return np.tanh(a*x**3)
        if mode==1: return np.tanh(a*x**3+b*x)
        if mode==2: return np.sin(a*x**3+b*x)
        if mode==3: return np.arcsinh(a*x**3+b*x)
        if mode==4: return np.tanh(a*x)+b*np.sin(1.7*x)
        if mode==5: return np.tanh(a*x**3)+b*np.tanh(1.7*x)
        if mode==6: return np.sin(a*x)+b*np.sin(1.7*x)
        if mode==7: return np.tanh(a*x)+b*np.tanh(1.7*x)
        if mode==8: return np.tanh(a*x)+b*x**3
        if mode==9: return np.tanh(a*x**3/(1+abs(b)*x*x))
        return np.tanh(a*x)+b*x**3/(1+x*x)
    def f(x,t): return np.c_[np.ones(len(x)),x,phi(x,t)]
    def r(t):
        F=f(u,t); H=np.linalg.pinv(F,rcond=1e-9)
        return ((F@(H@v)-v)/np.maximum(.04,1-(F*H.T).sum(1))[:,None]).ravel()
    fit=min((least_squares(r,t,bounds=([.05,-4],[8,4]),max_nfev=65) for t in ((.8,.15),(1.7,-.25),(3,.6))),key=lambda o:o.cost)
    th=fit.x; F=f(u,th)
    return f(x,th)@np.linalg.lstsq(F,v,rcond=1e-9)[0],np.mean(r(th)**2)**.5

def _scalar_features(b,kind,k):
    P=_polys(b,3); C=P[:,15:]; Q=P[:,5:15]
    if kind==0: return np.c_[P[:,:15],np.tanh(k*C)/k]
    if kind==1: return np.c_[P[:,:15],np.arcsinh(k*C)/k]
    if kind==2: return np.c_[P[:,:15],C/(1+k*C*C)]
    if kind==3: return np.c_[P[:,:5],np.tanh(k*Q)/k,np.tanh(k*C)/k]
    if kind==4: return np.c_[P[:,:15],np.tanh(k*C)/k,np.sin(k*Q)/k]
    if kind==5: return np.c_[P[:,:15],np.tanh(k*C)/k,np.tanh(k*_polys(b,4)[:,35:])/k]
    if kind==6: return np.c_[P[:,:15],np.tanh(k*C)/k,C]
    if kind==7: return np.c_[P[:,:5],Q,np.tanh(k*Q)/k,np.tanh(k*C)/k]
    if kind==8: return np.c_[P[:,:15],np.tanh(k*C)/k,_polys(np.tanh(k*b)/k,2)[:,1:]]
    i,j=np.triu_indices(4,1); S=(b[:,i]+b[:,j])**2; D=(b[:,i]-b[:,j])**2
    return np.c_[P[:,:15],np.tanh(k*C)/k,(Q[:,:,None]/(1+k*np.c_[S,D][:,None,:])).reshape(len(b),-1)]

def _scalar_fit(F,b,y,n,lam,scaling,link):
    A=F[:n]; scale=np.maximum(.25,A.std(0)) if scaling else 1.; G=(F-A.mean(0))/scale; T=G[:n]; mean=y.mean()
    if link in (1,2,-1,-2):
        c=max(abs(y))*({1:1.05,2:1.3,-1:1.15,-2:1.6}[link]); v=c*np.arctanh(y/c) if link>0 else c*np.tanh(y/c)
        mean=v.mean(); z=G[n:]@T.T@solve(T@T.T+lam*np.eye(n),v-mean,assume_a='pos')+mean
        return c*np.tanh(z/c) if link>0 else c*np.arctanh(np.clip(z/c,-.97,.97))
    if link==3:
        q=np.c_[b,b[:,np.arange(4)]**2,_polys(b,2)[:,5:]]; q=q/np.maximum(.3,q[:n].std(0)); X=np.c_[np.ones(n),T,-y[:,None]*q[:n]]
        reg=np.r_[1e-8,np.full(T.shape[1],lam),np.full(q.shape[1],lam*12+.02)]; X1=X/np.sqrt(reg); c=(X1.T@solve(X1@X1.T+np.eye(n),y,assume_a='pos'))/np.sqrt(reg)
        return (c[0]+G[n:]@c[1:1+T.shape[1]])/np.maximum(.4,1+q[n:]@c[1+T.shape[1]:])
    return G[n:]@T.T@solve(T@T.T+lam*np.eye(n),y-mean,assume_a='pos')+mean
'''.replace('\n    if link==3:', '\n    if link==3:'),globals())
_polys=old._polys
from sklearn.model_selection import KFold
from time import monotonic
start=monotonic(); tr=s['train']; y=tr['y']; u=old._inverse(s['panel_a'],tr['x']); panel=s['panel_b']; bs=[]
for k in range(11):
    b,e=_new_b(panel,u,k); bs.append((e,('new',k),b))
print('Calibration LOO:',[(round(e,6),tag) for e,tag,_ in sorted(bs,key=lambda a:a[0])],flush=True)
bs=sorted(bs,key=lambda a:a[0])[:4]+[(0,('old',None),old._panel_b(panel,u)),(0,('old',('csat','lin',[1.702,1.081])),old._calibrate_b(panel,u,'csat','lin',[1.702,1.081]))]
inner=list(KFold(4,shuffle=True,random_state=17).split(y[:52])); tail=(np.arange(52),np.arange(52,len(y))); records=[]
for _,tag,b in bs:
    for kind in range(10):
        for k in (.6,1.7,2.6,3.2,4.,5.,6.,8.,12.,18.):
            F=_scalar_features(b,kind,k)
            for lam in (.002,.008,.016,.024,.045,.08,.16,.35):
                for scale in (False,True):
                    for link in (0,1,2,-1,3):
                        a,h=tail; er=np.mean(abs(_scalar_fit(F,b,y[a],len(a),lam,scale,link)-y[h]))
                        if er<=.049: records.append((float(er),tag,kind,k,lam,scale,link))
print('Public candidates:',len(records),'minimum:',min((r[0] for r in records),default=None),'seconds:',round(monotonic()-start,2),flush=True)
eligible=[r for r in records if r[0]<=.0379]; pool=eligible or sorted(records,key=lambda r:r[0])[:20]; ranked=[]
for row in pool:
    er,tag,kind,k,lam,scale,link=row; b=next(b for _,tg,b in bs if tag==tg); F=_scalar_features(b,kind,k); es=[]
    for a,h in inner:
        idx=np.r_[a,h]; es.append(np.mean(abs(_scalar_fit(F[idx],b[idx],y[a],len(a),lam,scale,link)-y[h])))
    ranked.append((float(.6*np.mean(es)+.4*max(es)),float(max(es)),row))
if ranked:
    ranked.sort(key=lambda r:r[0] if eligible else (r[2][0]+.1*r[0])); score,worst,(er,tag,kind,k,lam,scale,link)=ranked[0];print('Selected: public',er,'worst internal',worst,'configuration',(tag,kind,k,lam,scale,link),flush=True)
    if er<.046845:
        import inspect
        helper='\n\n'.join(inspect.getsource(f) for f in (_new_b,_scalar_features,_scalar_fit)) if False else pathlib.Path(__file__).read_text().split("exec('''",1)[1].split("'''.replace",1)[0]
        out=source.replace('def run_node(inputs: dict) -> dict:',helper+'\n\ndef run_node(inputs: dict) -> dict:')
        if tag[0]=='new': cal=f'_new_b(s["panel_b"],w,{tag[1]})[0]'
        elif tag[1] is None: cal='_panel_b(s["panel_b"],w)'
        else: cal='_calibrate_b(s["panel_b"],w,*'+repr(tag[1])+')'
        out=out.replace('y=preds[0]; z=np.outer(preds[1],q)',f'b_scalar={cal}\n    Fs=_scalar_features(b_scalar,{kind},{k})\n    y=_scalar_fit(Fs,b_scalar,tr["y"],n,{lam},{scale},{link}); z=np.outer(preds[1],q)')
        if out!=source and 'b_scalar=' in out: compile(out,str(p),'exec');p.write_text(out)
