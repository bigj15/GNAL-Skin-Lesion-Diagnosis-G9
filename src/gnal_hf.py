"""Independent GNAL implementation. See README for assumptions, not official code."""
import argparse, json, math, os, random, time, warnings, hashlib
from tqdm.auto import tqdm
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T, models
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score

CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
@dataclass
class Config:
    data_root: str = '/kaggle/input'
    prepared_manifest: str = ''
    metadata_csv: str = ''
    labels_csv: str = ''
    output: str = '/kaggle/working/gnal_run'
    seed: int = 42
    fold: int = 0
    label_ratio: float = .2
    warmup_epochs: int = 20
    epochs: int = 100
    batch_size: int = 12
    unlabeled_ratio: int = 1
    workers: int = 2
    image_size: int = 224
    lr: float = .001
    lambda_n: float = .5
    lambda_g: float = .4
    eta: float = .3
    temperature: float = .1
    neighbor_threshold: float = .8
    max_neighbors: int = 32
    context_limit: int = 96
    gamma: float = 1.
    beta: float = .1
    confidence: float = .95
    hidden: int = 256
    pretrained: bool = False
    imagenet_normalize: bool = False
    amp: bool = True
    ablation: str = 'full'  # full, no_ncfr, no_rega, supervised
    max_steps: int = 0
    resume: str = ''
    tiny: bool = False  # synthetic smoke tests only
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def discover_data(c):
    root = Path(c.data_root)
    csvs = sorted(root.rglob('*.csv'))
    metadata = Path(c.metadata_csv) if c.metadata_csv else None
    labels = Path(c.labels_csv) if c.labels_csv else None
    if metadata is None:
        candidates = []
        for p in csvs:
            try:
                cols = pd.read_csv(p, nrows=1).columns
                if 'lesion_id' in cols and ('image_id' in cols or 'image' in cols): candidates.append(p)
            except Exception: pass
        if len(candidates) != 1:
            raise ValueError(f'Found {len(candidates)} lesion metadata CSVs. Set metadata_csv explicitly. Candidates: {candidates}')
        metadata = candidates[0]
    df = pd.read_csv(metadata).rename(columns={'image':'image_id'})
    if not {'lesion_id', 'image_id'}.issubset(df.columns):
        raise ValueError('Metadata must contain image_id (or image) AND lesion_id. Add HAM10000_metadata.csv; image-only splitting is deliberately disabled.')
    df['image_id'] = df.image_id.astype(str).map(lambda x: Path(x).stem)
    if 'dx' in df.columns:
        df['class'] = df.dx.astype(str).str.upper()
    else:
        if labels is None:
            hits = []
            for p in csvs:
                cols = pd.read_csv(p, nrows=1).columns
                if set(CLASSES).issubset(cols): hits.append(p)
            if len(hits) != 1: raise ValueError(f'Set labels_csv explicitly. Candidates: {hits}')
            labels = hits[0]
        lab = pd.read_csv(labels).rename(columns={'image':'image_id'})
        lab['image_id'] = lab.image_id.astype(str).map(lambda x: Path(x).stem)
        vals = lab[CLASSES].to_numpy(float)
        if not (np.isin(vals, [0,1]).all() and (vals.sum(1)==1).all()):
            raise ValueError('Expected one-hot ground truth, exactly one class per row.')
        lab['class'] = np.array(CLASSES)[vals.argmax(1)]
        df = df.merge(lab[['image_id','class']], on='image_id', how='left', validate='one_to_one')
        if df['class'].isna().any(): raise ValueError('Some metadata images have no corresponding ground-truth labels.')
    if df.image_id.duplicated().any() or df.lesion_id.isna().any():
        raise ValueError('Duplicate image IDs or missing lesion IDs.')
    if not set(df['class']).issubset(CLASSES): raise ValueError('Unknown class labels.')
    if (df.groupby('lesion_id')['class'].nunique() > 1).any(): raise ValueError('A lesion has conflicting diagnoses.')
    images = {}; duplicate_copies = 0
    def digest(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        return h.digest()
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() in ('.jpg','.jpeg','.png'):
            if p.stem in images:
                previous=Path(images[p.stem])
                if previous.stat().st_size!=p.stat().st_size or digest(previous)!=digest(p):
                    raise ValueError(f'Conflicting image copies for {p.stem}; attach one dataset copy only.')
                duplicate_copies+=1
                continue
            images[p.stem] = str(p)
    if duplicate_copies: print('Ignored byte-identical duplicate image copies:',duplicate_copies)
    df['path'] = df.image_id.map(images)
    if df.path.isna().any(): raise ValueError(f'Missing images: {df.loc[df.path.isna(), "image_id"].head().tolist()}')
    df['target'] = df['class'].map({x:i for i,x in enumerate(CLASSES)})
    print('Data:', len(df), 'images;', df.lesion_id.nunique(), 'lesions;', metadata)
    if len(df) != 10015: warnings.warn('Paper uses 10,015 images; this dataset has a different size.')
    return df.reset_index(drop=True)


def make_splits(df, c):
    if not 0 < c.label_ratio < 1: raise ValueError('label_ratio must be between 0 and 1.')
    if not 0 <= c.fold < 5: raise ValueError('fold must be 0..4')
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=c.seed)
    trval, test = list(outer.split(df, df.target, df.lesion_id))[c.fold]
    pool = df.iloc[trval]
    # One of eight inner folds = ~10% total validation; outer test = ~20%.
    inner = StratifiedGroupKFold(8, shuffle=True, random_state=c.seed+1)
    tri, vai = next(inner.split(pool, pool.target, pool.lesion_id))
    train, val, test = pool.iloc[tri].copy(), pool.iloc[vai].copy(), df.iloc[test].copy()
    for a,b in [(train,val),(train,test),(val,test)]:
        assert not set(a.lesion_id) & set(b.lesion_id)
    # Label complete lesions to avoid one labelled view effectively labelling another hidden view.
    groups = train.groupby('lesion_id').target.first().reset_index()
    if groups.target.value_counts().min() < 2: raise ValueError('Too few lesions per class to select labelled data.')
    chosen, _ = train_test_split(groups, train_size=c.label_ratio, stratify=groups.target, random_state=c.seed+2)
    train['labeled'] = train.lesion_id.isin(chosen.lesion_id)
    train = train.reset_index(drop=True)
    if train.loc[train.labeled,'target'].nunique() != 7: raise ValueError('Labelled set must contain all seven classes.')
    val['labeled'] = False; test['labeled'] = False
    train['split']='train'; val['split']='validation'; test['split']='test'
    manifest = pd.concat([train,val,test], ignore_index=True)
    manifest.to_csv(Path(c.output)/'split_manifest.csv', index=False)
    print('Split sizes:', len(train), len(val), len(test), '; labelled training images:', train.labeled.sum())
    return train, val.reset_index(drop=True), test.reset_index(drop=True)


def transform(c, kind):
    ops = [T.Resize((c.image_size,c.image_size))]
    if kind != 'eval': ops += [T.RandomHorizontalFlip(), T.RandomRotation(20)]
    if kind == 'strong': ops += [T.RandAugment(num_ops=2, magnitude=9)]
    ops += [T.ToTensor()]
    if c.imagenet_normalize: ops += [T.Normalize([.485,.456,.406],[.229,.224,.225])]
    return T.Compose(ops)

class Lesions(Dataset):
    def __init__(self, frame, c, kind='eval', hide=False):
        self.df=frame; self.weak=transform(c,kind); self.strong=transform(c,'strong'); self.kind=kind; self.hide=hide
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.path) as im: im=im.convert('RGB')
        x=self.weak(im)
        return x, self.strong(im) if self.kind=='unlabeled' else x, (-1 if self.hide else int(r.target)), int(r.name)

class Encoder(nn.Module):
    def __init__(self,c):
        super().__init__()
        if c.tiny:
            self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
            self.dim=16
        else:
            m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT if c.pretrained else None)
            self.dim=m.fc.in_features; m.fc=nn.Identity(); self.net=m
    def forward(self,x): return self.net(x)

class GNAL(nn.Module):
    def __init__(self,c):
        super().__init__(); self.encoder=Encoder(c); d=self.encoder.dim
        self.head=nn.Linear(d,7); self.g1=nn.Linear(d,c.hidden,bias=False); self.g2=nn.Linear(c.hidden,7,bias=False)
    def graph(self,z,a):
        # Works with [nodes,d] or batched [batch,nodes,d].
        return a @ self.g2(F.relu(a @ self.g1(z)))


def normalize_adj(a):
    a=a+torch.eye(a.shape[-1],device=a.device)
    inv=a.sum(-1).clamp_min(1e-8).rsqrt()
    return inv[..., :, None]*a*inv[...,None,:]


def neighbor_query(z, bank, ids, c):
    logits=F.normalize(z.float(),dim=-1) @ bank.T / c.temperature
    logits[torch.arange(len(z),device=z.device),ids]=-float('inf')
    width=min(c.max_neighbors or len(bank)-1,len(bank)-1)
    values, indices=logits.detach().topk(width,dim=1)
    mask=(values-values[:,:1]).exp() >= c.neighbor_threshold
    return [set(row[keep].cpu().tolist()) for row,keep in zip(indices,mask)], logits

@torch.no_grad()
def refresh_neighbors(bank,c):
    result=[]
    for start in tqdm(range(0,len(bank),128),desc='Refreshing neighborhoods',unit='chunk',leave=False):
        ids=torch.arange(start,min(start+128,len(bank)),device=bank.device)
        ns,_=neighbor_query(bank[ids],bank,ids,c); result.extend(ns)
    return result

def jaccard(a,b): return len(a & b)/max(1,len(a | b))

def graph_adjacency(node_ids, neighborhoods, labeled, device):
    n=len(node_ids); a=np.zeros((n,n),np.float32)
    for i,u in enumerate(node_ids):
        for j,v in enumerate(node_ids):
            if i!=j and (v in neighborhoods[u] or u in neighborhoods[v]):
                a[i,j]=1. if labeled[u] and labeled[v] else jaccard(neighborhoods[u], neighborhoods[v])
    return torch.tensor(a,device=device)

def ncfr_loss(ids, logits, neighborhoods, labeled):
    terms=[]; lp=logits.log_softmax(1)
    for row,i in enumerate(ids.tolist()):
        if labeled[i]: continue
        ns=sorted(neighborhoods[i])
        if ns:
            w=torch.tensor([jaccard(neighborhoods[i],neighborhoods[j]) for j in ns],device=logits.device)
            terms.append(-(w*lp[row,ns]).mean())
    return torch.stack(terms).mean() if terms else logits[torch.isfinite(logits)].sum()*0


def graph_train(model,z,ids,bank,neighbors,labeled,known_y,c,augment):
    z=F.normalize(z.float(),dim=1)
    unique, inverse=torch.unique(ids,sorted=True,return_inverse=True)
    # Weighted sampling may repeat images. Preserve one graph node per image; average their live features.
    live=torch.zeros((len(unique),z.shape[1]),device=z.device,dtype=z.dtype).index_add(0,inverse,z)
    counts=torch.bincount(inverse,minlength=len(unique)).to(z.dtype)
    live=live/counts[:,None]
    active=unique.tolist(); used=set(active)
    contexts=sorted(set().union(*(neighbors[i] for i in active))-used)
    if len(contexts)>c.context_limit:
        # Deterministic ranking by aggregate similarity to current batch.
        ctx=torch.tensor(contexts,device=z.device)
        ranks=(bank[ctx] @ F.normalize(live.detach().float(),dim=1).T).amax(1)
        contexts=ctx[ranks.topk(c.context_limit).indices].tolist()
    nodes=active+contexts
    features=torch.cat([live.float(),bank[contexts]],0)
    a=graph_adjacency(nodes,neighbors,labeled,z.device)
    h=model.graph(features,normalize_adj(a)); hp=h.softmax(-1)
    idx_l=[j for j,i in enumerate(active) if labeled[i]]
    idx_u=[j for j,i in enumerate(active) if not labeled[i]]
    gl=F.cross_entropy(h[idx_l],known_y[unique[idx_l]])
    lg=h.sum()*0
    if augment and idx_u:
        number=max(1,math.ceil(c.eta*len(idx_u))) if c.eta>0 else 0
        picks=random.sample(idx_u,min(number,len(idx_u)))
        if picks:
            li=torch.tensor(random.choices(idx_l,k=len(picks)),device=z.device)
            ui=torch.tensor(picks,device=z.device)
            sub=[]
            for j,i in enumerate(nodes):
                positions=[k for k,v in enumerate(nodes) if v in neighbors[i]]+[j]
                sub.append(features[positions].mean(0))
            sub=torch.stack(sub)
            theta0=.5*torch.exp(-c.gamma*(sub[li]-sub[ui]).square().sum(1))
            entropy=-(hp.detach()*hp.detach().clamp_min(1e-8).log()).sum(1)
            theta=(theta0+c.beta*(entropy[li]-entropy[ui])/math.log(7)).clamp(0,1).detach()
            mixed=theta[:,None]*features[li]+(1-theta[:,None])*features[ui]
            aug_a=torch.zeros((len(nodes)+len(picks),)*2,device=z.device); aug_a[:len(nodes),:len(nodes)]=a
            for q,(l,u) in enumerate(zip(li.tolist(),ui.tolist())):
                # New node inherits union of parent neighborhoods and links to both parents.
                connected=((a[l]>0)|(a[u]>0)).float(); connected[l]=1; connected[u]=1
                aug_a[len(nodes)+q,:len(nodes)]=connected
                aug_a[:len(nodes),len(nodes)+q]=connected
            final=model.graph(torch.cat([features,mixed]),normalize_adj(aug_a))
            target=theta[:,None]*F.one_hot(known_y[unique[li]],7).float()+(1-theta[:,None])*final[ui].detach().softmax(-1)
            lg=-(target*final[len(nodes):].log_softmax(-1)).sum(1).mean()
            gl=F.cross_entropy(final[idx_l],known_y[unique[idx_l]])
    return gl,lg


def loader(frame,c,kind,balanced=False,hide=False):
    ds=Lesions(frame,c,kind,hide)
    sampler=None
    if balanced:
        counts=frame.target.value_counts()
        sampler=WeightedRandomSampler([1/counts[y] for y in frame.target],len(frame),replacement=True)
    return DataLoader(ds,batch_size=c.batch_size*(c.unlabeled_ratio if hide else 1),sampler=sampler,
                      shuffle=(kind!='eval' and sampler is None),num_workers=c.workers,pin_memory=c.device.startswith('cuda'),drop_last=False)

def endlessly(dl):
    while True:
        yield from dl

@torch.no_grad()
def fill_bank(model,dl,bank,c):
    model.eval()
    for x,_,_,ids in tqdm(dl,desc='Updating feature memory',unit='batch',leave=False):
        bank[ids.to(c.device)]=F.normalize(model.encoder(x.to(c.device)).float(),dim=1)

@torch.no_grad()
def predict(model,dl,bank,neighbors,labeled,c):
    """Each held-out query uses its own training-only context, never another held-out sample."""
    model.eval(); ys=[]; probs=[]; image_ids=[]
    for x,_,y,ids in tqdm(dl,desc='Evaluating images',unit='batch',leave=False):
        z=model.encoder(x.to(c.device)).float()
        if c.ablation=='supervised': p=model.head(z).softmax(-1)
        else:
            logits=F.normalize(z,dim=1) @ bank.T/c.temperature
            width=min(c.max_neighbors or len(bank),len(bank))
            vals, ni=logits.topk(width,dim=1)
            ps=[]
            for q in range(len(z)):
                ctx=ni[q][(vals[q]-vals[q,0]).exp()>=c.neighbor_threshold].tolist()
                ctx=ctx[:c.context_limit]
                features=torch.cat([F.normalize(z[q:q+1],dim=1),bank[ctx]])
                a=torch.zeros((len(ctx)+1,)*2,device=c.device)
                a[1:,1:]=graph_adjacency(ctx,neighbors,labeled,c.device)
                # Query receives messages from training nodes, no messages back into reference context.
                a[0,1:]=1.
                ps.append(model.graph(features,normalize_adj(a))[0].softmax(-1))
            p=torch.stack(ps)
        ys.extend(y.tolist()); probs.extend(p.cpu().tolist()); image_ids.extend(ids.tolist())
    return np.array(ys),np.array(probs),image_ids


def metrics(y,p):
    pred=p.argmax(1); cm=confusion_matrix(y,pred,labels=range(7))
    tp=cm.diagonal(); fn=cm.sum(1)-tp; fp=cm.sum(0)-tp; tn=cm.sum()-tp-fn-fp
    result={'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,labels=range(7),average='macro',zero_division=0)),
            'macro_sensitivity':float(np.mean(tp/np.maximum(tp+fn,1))), 'macro_specificity':float(np.mean(tn/np.maximum(tn+fp,1)))}
    try: result['macro_auc']=float(roc_auc_score(y,p,multi_class='ovr',labels=range(7),average='macro'))
    except ValueError: result['macro_auc']=None
    onehot=np.eye(7)[y]; result['brier_multiclass']=float(np.square(p-onehot).sum(1).mean())
    conf=p.max(1); correct=(pred==y); ece=0.
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(conf>lo)&(conf<=hi)
        if mask.any(): ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    result['ece_toplabel_15bins']=float(ece)
    return result,cm


def save_plots(out,history,y,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
    if history:
        h=pd.DataFrame(history); fig,ax=plt.subplots(1,2,figsize=(11,4))
        ax[0].plot(h.epoch,h.loss); ax[0].set(title='Training loss',xlabel='Epoch')
        ax[1].plot(h.epoch,h.val_macro_f1,label='Validation macro F1'); ax[1].plot(h.epoch,h.val_accuracy,label='Validation accuracy'); ax[1].legend()
        fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=160); plt.close(fig)
    _,cm=metrics(y,p); fig,ax=plt.subplots(figsize=(8,7)); ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(ax=ax,colorbar=False)
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6))
    for j,name in enumerate(CLASSES):
        if len(np.unique(y==j))==2:
            fpr,tpr,_=roc_curve(y==j,p[:,j]); ax.plot(fpr,tpr,label=f'{name}: {auc(fpr,tpr):.3f}')
    ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Test one-vs-rest ROC'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'roc_curves.png',dpi=160); plt.close(fig)


def run(c):
    if c.temperature<=0 or not 0<c.neighbor_threshold<=1: raise ValueError('Invalid temperature or neighbor threshold.')
    if c.context_limit<1: raise ValueError('context_limit must be positive')
    if c.ablation not in ['full','no_ncfr','no_rega','supervised']: raise ValueError('Invalid ablation')
    if c.pretrained and not c.imagenet_normalize: warnings.warn('ImageNet weights normally use imagenet_normalize=True.')
    seed_all(c.seed); out=Path(c.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not c.resume:
        raise FileExistsError('Output already contains a run. Set resume to last.pt or choose a new output directory.')
    resume=None
    if c.resume:
        resume=torch.load(c.resume,map_location='cpu',weights_only=False)
        old=resume['config']
        for key in asdict(c):
            if key not in ['resume','output','workers','device','epochs','max_steps','amp','data_root','metadata_csv','labels_csv','prepared_manifest'] and old[key]!=getattr(c,key):
                raise ValueError(f'Resume config mismatch: {key}')
    (out/'config.json').write_text(json.dumps(asdict(c),indent=2))
    import torchvision, sklearn, platform
    environment={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
                 'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
                 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    if c.prepared_manifest:
        from hf_data import validate_manifest
        prepared=Path(c.prepared_manifest)
        info=json.loads(prepared.with_name('preparation.json').read_text())
        for key in ['seed','fold','label_ratio']:
            if info[key]!=getattr(c,key): raise ValueError('Prepared dataset config mismatch: '+key)
        if info['classes']!=CLASSES: raise ValueError('Class ordering mismatch')
        frame=validate_manifest(pd.read_csv(prepared))
        frame.to_csv(out/'split_manifest.csv',index=False)
        train=frame[frame.split=='train'].reset_index(drop=True)
        val=frame[frame.split=='validation'].reset_index(drop=True)
        test=frame[frame.split=='test'].reset_index(drop=True)
    else:
        df=discover_data(c); train,val,test=make_splits(df,c)
    if resume and resume['train_ids']!=train.image_id.tolist(): raise ValueError('Resume split/data order mismatch.')
    # Runtime target table contains -1 for all unlabelled images.
    labeled=train.labeled.to_numpy(); known_y=torch.tensor(np.where(labeled,train.target,-1),device=c.device)
    labeled_dl=loader(train[train.labeled],c,'weak',balanced=True)
    unlabeled_dl=loader(train[~train.labeled],c,'unlabeled',hide=True)
    bank_dl=loader(train,c,'eval'); val_dl=loader(val,c,'eval'); test_dl=loader(test,c,'eval')
    model=GNAL(c).to(c.device)
    opt=torch.optim.Adam(model.parameters(),lr=c.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=20,gamma=.1)
    enabled=c.amp and c.device.startswith('cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=enabled)
    bank=torch.zeros((len(train),model.encoder.dim),device=c.device)
    start=0; best=-1.; history=[]
    if resume:
        model.load_state_dict(resume['model']); opt.load_state_dict(resume['optimizer']); scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler']); start=resume['epoch']+1; best=resume['best']; history=resume['history']
        random.setstate(resume['python_rng']); np.random.set_state(resume['numpy_rng']); torch.set_rng_state(resume['torch_rng'])
        if torch.cuda.is_available() and resume['cuda_rng'] is not None: torch.cuda.set_rng_state_all(resume['cuda_rng'])
    print('Initializing training-only feature memory...',flush=True)
    fill_bank(model,bank_dl,bank,c); neighbors=refresh_neighbors(bank,c)
    total=c.warmup_epochs+c.epochs
    print('Device:',c.device,'; visible GPUs:',torch.cuda.device_count(),'; using one GPU intentionally')
    for epoch in range(start,total):
        began=time.time(); warm=epoch<c.warmup_epochs
        ul=endlessly(unlabeled_dl); ll=endlessly(labeled_dl)
        steps=len(labeled_dl) if warm or c.ablation=='supervised' else max(len(labeled_dl),len(unlabeled_dl))
        if c.max_steps: steps=min(steps,c.max_steps)
        model.train(); sums=np.zeros(5); accepted=0.; u_count=0
        print(f'Epoch {epoch+1}/{total} | {"warmup" if warm else "main training"} | {steps} batches',flush=True)
        progress=tqdm(range(steps),desc=f'Epoch {epoch+1}/{total}',unit='batch')
        for step in progress:
            xl,_,yl,il=next(ll); xl=xl.to(c.device); yl=yl.to(c.device); il=il.to(c.device)
            use_u=not warm and c.ablation!='supervised'
            if use_u:
                xu,xs,_,iu=next(ul); xu=xu.to(c.device); xs=xs.to(c.device); iu=iu.to(c.device)
                images=torch.cat([xl,xu,xs]); ids=torch.cat([il,iu])
            else: images=xl; ids=il
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(c.device).type,enabled=enabled):
                feats=model.encoder(images); logits=model.head(feats)
                base=F.cross_entropy(logits[:len(xl)],yl); unsup=base*0
                if use_u:
                    nu=len(xu); weak=logits[len(xl):len(xl)+nu]; strong=logits[-nu:]
                    conf,pseudo=weak.detach().softmax(-1).max(-1); mask=conf>=c.confidence
                    unsup=(F.cross_entropy(strong,pseudo,reduction='none')*mask).mean()
                    accepted+=mask.sum().item(); u_count+=nu
                    z=feats[:len(xl)+nu]
                else: z=feats
            # Global similarity and graph operations stay FP32 for numerical stability.
            with torch.autocast(device_type=torch.device(c.device).type,enabled=False):
                z=z.float(); current,sim=neighbor_query(z,bank,ids,c)
                for i,ns in zip(ids.tolist(),current): neighbors[i]=ns
                if c.ablation=='supervised': gl=base.float(); ln=z.sum()*0; lg=ln
                else:
                    gl,lg=graph_train(model,z,ids,bank,neighbors,labeled,known_y,c,use_u and c.ablation!='no_rega')
                    ln=ncfr_loss(ids,sim,neighbors,labeled) if use_u and c.ablation!='no_ncfr' else z.sum()*0
                loss=.5*(base.float()+gl)+unsup.float()+c.lambda_n*ln+c.lambda_g*lg
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss; checkpoint from previous epoch is retained.')
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                # Average duplicate entries deterministically.
                unique,inv=torch.unique(ids,return_inverse=True)
                emb=torch.zeros((len(unique),z.shape[1]),device=c.device).index_add_(0,inv,z.detach())
                bank[unique]=F.normalize(emb,dim=1)
            sums+=np.array([loss.item(),base.item(),unsup.item(),ln.item(),lg.item()])
            progress.set_postfix(loss=f'{loss.item():.4f}',ncfr=f'{ln.item():.4f}',graph=f'{lg.item():.4f}',refresh=(step%10==0))
        scheduler.step()
        # Synchronize reference bank with current encoder before validation and checkpointing.
        fill_bank(model,bank_dl,bank,c); neighbors=refresh_neighbors(bank,c)
        print('Checking validation performance...',flush=True)
        vy,vp,_=predict(model,val_dl,bank,neighbors,labeled,c); vm,_=metrics(vy,vp)
        row=dict(epoch=epoch+1,stage='warmup' if warm else 'main',seconds=time.time()-began,loss=sums[0]/steps,
                 base=sums[1]/steps,unsupervised=sums[2]/steps,ncfr=sums[3]/steps,graph=sums[4]/steps,
                 pseudo_acceptance=accepted/max(u_count,1),val_macro_f1=vm['macro_f1'],val_accuracy=vm['accuracy'],lr=opt.param_groups[0]['lr'])
        history.append(row); pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        # Select only main-training epochs for a GNAL best model; warmup cannot win against main stage.
        eligible=not warm or c.epochs==0
        improved=eligible and vm['macro_f1']>best
        if improved: best=vm['macro_f1']
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,
                   best=best,history=history,config=asdict(c),train_ids=train.image_id.tolist(),classes=CLASSES,
                   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(state,out/'last.tmp'); os.replace(out/'last.tmp',out/'last.pt')
        if improved: torch.save(state,out/'best.pt')
        callback=globals().get('CHECKPOINT_CALLBACK')
        if callback is None: raise RuntimeError('Cloud backup is not configured. Run the HF backup setup cell first.')
        callback(out,epoch+1,improved,history)
    # Never inspect test metrics during training or use them for checkpoint selection.
    if not (out/'best.pt').exists():
        if c.resume and Path(c.resume).with_name('best.pt').exists():
            best_path=Path(c.resume).with_name('best.pt')
        else: raise RuntimeError('No best checkpoint here. Copy best.pt alongside last.pt when resuming to a new output directory.')
    else: best_path=out/'best.pt'
    selected=torch.load(best_path,map_location=c.device,weights_only=False); model.load_state_dict(selected['model'])
    fill_bank(model,bank_dl,bank,c); neighbors=refresh_neighbors(bank,c)
    print('Evaluating the selected checkpoint on the held-out test split...',flush=True)
    y,p,indices=predict(model,test_dl,bank,neighbors,labeled,c)
    result,cm=metrics(y,p); result['best_epoch']=selected['epoch']+1
    (out/'test_metrics.json').write_text(json.dumps(result,indent=2))
    report=classification_report(y,p.argmax(1),labels=range(7),target_names=CLASSES,zero_division=0,output_dict=True)
    pd.DataFrame(report).T.to_csv(out/'classification_report.csv')
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/'confusion_matrix.csv')
    pred=pd.DataFrame(p,columns=['prob_'+n for n in CLASSES]); pred.insert(0,'image_id',test.iloc[indices].image_id.to_numpy())
    pred['true_class']=np.array(CLASSES)[y]; pred['predicted_class']=np.array(CLASSES)[p.argmax(1)]; pred.to_csv(out/'test_predictions.csv',index=False)
    save_plots(out,history,y,p); print('TEST (held out):',json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True)
    args=ap.parse_args(); run(Config(**json.loads(Path(args.config).read_text())))
