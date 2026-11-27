exec(open('/tmp/rheo_cv.py').read().split('for lab in (*np.unique(labs), -1):')[0])
import json
from scipy.optimize import least_squares
for sw in json.loads(s._ARCHIVE.read_text()):
    ls,c,y=(np.asarray(sw[k]) for k in ('lab_id','cmd_stress','true_stress'))
    vals=[]
    for j in range(4):
        a,b=c[ls==j],y[ls==j]; k=~s._spikes(a,b); vals.append(np.round(s._line(a[k],b[k]),6).tolist())
    print('sweep',sw['episode_short_name'],vals)

def fit(a,yc,yr,robust):
    y=np.r_[yc.ravel(),yr]
    def profile(u, ret=False):
        c,r=s._design(np.exp(u),dt,t,*a)
        A=np.vstack((c.reshape(-1,9),r))
        w=np.linalg.lstsq(A,y,rcond=None)[0]; e=A@w-y
        if robust:
            scale=np.r_[np.full(yc.size,.025),np.full(yr.size,.065)]
            for j in range(18):
                q=(1+(e/scale)**2)**(-.25)
                w=np.linalg.lstsq(A*q[:,None],y*q,rcond=None)[0]; e=A@w-y
            obj=e*np.sqrt(2/(1+np.sqrt(1+(e/scale)**2)))
        else: obj=e
        return w if ret else obj
    u=least_squares(profile,np.log([.62,3.8]),bounds=(np.log(.025),np.log(100))).x
    return np.exp(u),profile(u,True)

for label,ar,rob in [('gated LS',args,False),('gated robust',args,True),('all log robust',(*args[:-1],np.ones(n,bool)),True)]:
    print(label)
    for lab in (*np.unique(labs),-1):
        tr=(labs!=lab); te=(labs==lab) if lab!=-1 else tr
        tau,w=fit(tuple(a[tr] for a in ar),yc[tr],yr[tr],rob)
        c,r=s._design(tau,dt,t,*(a[te] for a in ar)); ec=c@w-yc[te]; er=r@w-yr[te]
        print(lab,'tau',tau.round(4),'curve abs quantiles',np.quantile(np.abs(ec),[.5,.8,.9,.95,.99]).round(4),'rate abs quantiles',np.quantile(np.abs(er),[.5,.8,.9,.95]).round(4))
        if lab==-1:
            print('weights',w.round(5))
            for k in (False,True):
                m=args[-1]==k
                print('plateau',k,'curve medians by t',np.median(np.abs(ec[m]),axis=0).round(4),'rate median abs',np.median(np.abs(er[m])).round(4))
            if label=='gated robust':
                for i in np.argsort(np.median(np.abs(ec),axis=1))[-7:]: print('row',i,'lab',labs[i],'damage gate',args[-1][i],'curve residual',ec[i].round(3),'rate residual',round(er[i],3))
                print('count ec > thresholds',[(x,int((np.abs(ec)>x).sum())) for x in [.025,.05,.1,.2,.5,1,3]])
