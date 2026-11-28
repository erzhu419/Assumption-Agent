def _scalar(q, b, c, y, k):
    n = len(y)
    one = np.ones(len(q))
    def designs():
        for a in (q, b):
            scores = list(a.T)
            scores += [a[:,i] + s*a[:,j] for i in range(4) for j in range(i+1,4) for s in (-1,1)]
            for h in scores:
                for strength in (k, 1.25):
                    g = 1/(1 + strength*h*h)
                    bg = b*g[:,None]
                    yield np.c_[one, bg, c]
                    yield np.c_[one, bg, c, bg*h[:,None]**3]
                    yield np.c_[one, bg, c, h, h*g, h*h*g]
                    yield np.c_[one, bg, c, b]
                    yield np.c_[one, bg, c, h**3*g]
                    yield np.c_[one, bg, c, h**3]
                    yield np.c_[one, bg, c, g]
                    yield np.c_[one, bg, c*g[:,None]]
        yield np.c_[one,b,c]
    results = []
    for a in designs():
        x = a[:n]
        u,s,vt = np.linalg.svd(x, full_matrices=False)
        keep = s > s[0]*1e-9
        u = u[:,keep]
        d = u.T@y
        e = abs(y-u@d)/np.maximum(1-np.sum(u*u,axis=1),.05)
        score = e.mean()
        results.append((score, a[n:] @ (vt[keep].T @ (d/s[keep]))))
    results.sort(key=lambda t:t[0])
    best = results[0][0]
    sel = [(s,p) for s,p in results[:8] if s < best + .006 or s == best]
    w = np.exp(-(np.array([s for s,p in sel])-best)/.0015)
    return np.average([p for s,p in sel], axis=0, weights=w)
