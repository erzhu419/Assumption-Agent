import numpy as np
from scipy.linalg import eigh_tridiagonal

def build(L,k,v,rg,d,mode=0,bound=True):
    h=L/len(k); diff=2*k[:-1]*k[1:]/(k[:-1]+k[1:])/h**2
    main=d.copy(); main[:-1]+=diff; main[1:]+=diff
    main[[0,-1]]+=rg*k[[0,-1]]/(h*(rg*.5*h+k[[0,-1]]))
    if mode in (0,1):
        pos=v.clip(0)/h; neg=(-v).clip(0)/h
        lo=-diff-pos[1:]; hi=-diff-neg[:-1]; main+=pos+neg
        if not bound: main[0]-=pos[0]; main[-1]-=neg[-1]
    elif mode in (2,3):
        vf=(v[1:]+v[:-1])/(2*h); pos=vf.clip(0); neg=(-vf).clip(0)
        lo=-diff-pos; hi=-diff-neg
        if mode==2: main[:-1]+=pos; main[1:]+=neg
        else: main[1:]+=pos; main[:-1]+=neg
        if bound: main[0]+=abs(v[0])/h; main[-1]+=abs(v[-1])/h
    eig,Q=eigh_tridiagonal(main,-np.sqrt(lo*hi)); scale=np.r_[1.,np.cumprod(np.sqrt(lo/hi))]; return eig,scale[:,None]*Q,Q.T/scale

def forward(s,mode,bound):
    V=s['V_reaction_2d']; dx=V.mean(axis=1); dy=V.mean(axis=0)-V.mean()
    (a,X,Ui),(b,Y,Vi)=[build(s['L'+p],s['kappa_'+p],s['vel_'+p],s['robin_'+p],d,mode,bound) for p,d in zip('xy',(dx,dy))]
    src=Ui@s['source_modes']@Vi.T
    u=X@(src[None,...]/(a[:,None]+b[None,:]+s['mu_shifts'][:,None,None,None]))@Y.T
    i,j=s['sensor_ij'].T
    return u[:,:,i,j].transpose(0,2,1),u[-1]

if __name__=='__main__':
    import pickle
    c=pickle.load(open('/holdout_task/public_beta_w02_s00.pkl','rb'))[0];s,g=c['study'],c['gt']; z=s['Z_nuisance']; ridge=s['nuisance_ridge']; a,d,_=np.linalg.svd(z,full_matrices=False); P=a@np.linalg.solve(z.T@z+ridge*np.eye(len(d)),np.zeros((len(d),1))) if False else z@np.linalg.solve(z.T@z+ridge*np.eye(len(d)),z.T)
    for mode in (0,2,3):
        for bound in (False,True):
            rows,end=forward(s,mode,bound)
            f=np.linalg.norm(g['w_opt']@end.reshape(len(end),-1)-g['terminal_field_A'].ravel())/np.linalg.norm(g['terminal_field_A'])
            for ordering,D in [('shift',rows.reshape(-1,rows.shape[-1])),('sensor',rows.transpose(1,0,2).reshape(-1,rows.shape[-1]))]:
                ratios=[np.linalg.norm(m-g['S_tilde'])/np.linalg.norm(g['S_tilde']) for m in (D,D-P@D,D-a@((1-np.sqrt(ridge/(d*d+ridge)))[:,None]*(a.T@D)))]; print(mode,bound,ordering,'design(raw,res,sqrt)',np.round(ratios,9),'terminal',f)
