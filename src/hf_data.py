"""Hugging Face transport for a prepared GNAL dataset; no token is stored."""
import hashlib, io, json, re
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import HfApi, hf_hub_download

CLASSES = ['AKIEC','BCC','BKL','DF','MEL','NV','VASC']
FEATURES = Features({'image':HFImage(decode=False),'image_id':Value('string'),
    'lesion_id':Value('string'),'class_name':Value('string'),'target':Value('int64'),
    'labeled':Value('bool'),'sha256':Value('string')})

def validate_manifest(frame, require_paths=True):
    f=frame.copy()
    required={'image_id','lesion_id','target','labeled','split'}
    if require_paths: required.add('path')
    if not required.issubset(f.columns): raise ValueError(f'Missing columns: {required-set(f.columns)}')
    if f[list(required)].isna().any().any(): raise ValueError('Missing manifest values')
    if set(f.split)!= {'train','validation','test'}: raise ValueError('Expected three splits')
    if f.image_id.duplicated().any(): raise ValueError('Duplicate image IDs across splits')
    if f.groupby('lesion_id').split.nunique().max()!=1: raise ValueError('Lesion leakage between splits')
    if f.groupby('lesion_id').target.nunique().max()!=1: raise ValueError('Inconsistent lesion diagnoses')
    if not set(f.target.astype(int)).issubset(range(7)): raise ValueError('Invalid class IDs')
    vals=f.labeled.astype(str).str.lower()
    if not vals.isin(['true','false']).all(): raise ValueError('labeled must be boolean')
    f['labeled']=vals.eq('true')
    if f.loc[f.split!='train','labeled'].any(): raise ValueError('Held-out rows cannot be training-labelled')
    tr=f[f.split=='train']
    if tr.groupby('lesion_id').labeled.nunique().max()!=1: raise ValueError('Label flag differs among views of a lesion')
    if tr.loc[tr.labeled,'target'].nunique()!=7 or (~tr.labeled).sum()==0: raise ValueError('Need labelled samples of all seven classes and unlabelled samples')
    if require_paths and not f.path.map(lambda p:Path(p).is_file()).all(): raise ValueError('Some prepared images are missing')
    return f


def build_dataset(manifest, destination, resize_to=224):
    """Keep source images untouched; export RGB JPEG copies or original bytes."""
    frame=validate_manifest(manifest); dest=Path(destination); dest.mkdir(parents=True,exist_ok=True)
    groups={}
    for split in ['train','validation','test']:
        rows=[]
        for r in frame[frame.split==split].itertuples():
            if not re.fullmatch(r'[A-Za-z0-9_-]+',str(r.image_id)): raise ValueError('Unsafe image ID')
            if resize_to:
                path=dest/(r.image_id+'.jpg')
                with PILImage.open(r.path) as im:
                    im.convert('RGB').resize((resize_to,resize_to),PILImage.Resampling.BILINEAR).save(path,'JPEG',quality=95)
            else:
                path=Path(r.path)
                with PILImage.open(path) as im: im.verify()
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({'image':{'path':str(path),'bytes':None},'image_id':r.image_id,
                         'lesion_id':str(r.lesion_id),'class_name':CLASSES[int(r.target)],
                         'target':int(r.target),'labeled':bool(r.labeled),'sha256':digest})
        groups[split]=Dataset.from_list(rows,features=FEATURES)
    return DatasetDict(groups)


def dataset_card(info):
    return '''---
license: cc-by-nc-4.0
task_categories:
- image-classification
language:
- en
tags:
- medical
- skin-lesion
- semi-supervised
- ham10000
---
# HAM10000: prepared GNAL experiment

This is a third-party preparation of HAM10000 / ISIC 2018 Task 3 for an independent GNAL implementation. It is not an official split or the GNAL author's release. Research/education experiment, not a clinical diagnostic service.

## Original data and attribution
HAM10000 Dataset: (c) ViDIR Group, Department of Dermatology, Medical University of Vienna. Original dataset: https://doi.org/10.7910/DVN/DBW86T .
Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161 .
ISIC challenge: Codella et al., Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration. https://arxiv.org/abs/1902.03368 .
Original data terms: https://challenge.isic-archive.com/data/ . Dataset license: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/ . Original attribution and noncommercial restriction remain applicable. This preparation claims no ownership of the original images.

## Changes made
Images are '''+('RGB-resized to '+str(info['resize_to'])+' square pixels and JPEG-encoded at quality 95.' if info['resize_to'] else 'stored as original image bytes without resizing.')+'''
Metadata is reduced to image/lesion IDs, diagnosis, split membership, label availability and image checksum. No local filesystem paths or credentials are published.

## Splits and labels
Five-fold stratified lesion-group splitting supplies one held-out outer fold, with a grouped inner validation partition (approximately 70/10/20 train/validation/test). All views of one lesion stay in the same split. Training labels are selected by complete lesion, stratified by diagnosis. The label budget is approximate by image count. This repo contains one fold and one label budget.

`target` and `class_name` contain the original ground truth for auditing. For training rows with `labeled=False`, the training notebook hides targets and never uses them for sampling or losses. Validation/test labels are used only for evaluation. Never infer training label availability from whether the public target column exists.

Class order: '''+json.dumps(CLASSES)+'''. `image` contains embedded image bytes in Parquet shards; no original download is needed by the training notebook.

## Preparation settings
```json
'''+json.dumps(info,indent=2)+'''
```

## Reference method
Chenglin Yu, Graph-enhanced neighborhood association learning model for semi-supervised skin lesion diagnosis (2026). https://doi.org/10.1016/j.bspc.2026.109886 .
Training is an independent implementation with documented graph and optimization assumptions; no reproduction accuracy is claimed by publishing this dataset.
'''


def publish_dataset(dataset,manifest,info,repo_id,token,allow_update=False):
    """Explicit public publication. Token is only passed in memory to Hub calls."""
    if '/' not in repo_id or repo_id.startswith('YOUR_'): raise ValueError('Set REPO_ID to your actual username/dataset-name')
    api=HfApi(token=token)
    api.create_repo(repo_id=repo_id,repo_type='dataset',private=False,exist_ok=True)
    if api.repo_info(repo_id,repo_type='dataset').private:
        raise ValueError('Existing repo is private; choose a new public dataset repo.')
    existing=set(api.list_repo_files(repo_id,repo_type='dataset'))-{'.gitattributes'}
    if existing and not allow_update: raise ValueError('Repo already contains files. Choose a new name, or set ALLOW_UPDATE=True to update this prepared dataset intentionally.')
    dataset.push_to_hub(repo_id,private=False,token=token,max_shard_size='200MB',embed_external_files=True,
                        commit_message='Publish lesion-separated HAM10000 experiment')
    # Preserve HF-generated dataset metadata in README; add our attribution after its YAML front matter.
    readme=Path(hf_hub_download(repo_id,'README.md',repo_type='dataset',token=token)).read_text()
    card=dataset_card(info)
    old_yaml=readme.split('---',2)[1] if readme.startswith('---') else ''
    new_yaml=card.split('---',2)[1]
    # DatasetDict writes configs/dataset_info here; preserve those blocks and avoid duplicate license keys.
    import yaml
    merged=yaml.safe_load(old_yaml) or {}; merged.update(yaml.safe_load(new_yaml) or {})
    body=card.split('---',2)[2]
    final='---\n'+yaml.safe_dump(merged,sort_keys=False)+'---\n'+body
    public_manifest=manifest[['image_id','lesion_id','target','labeled','split']].to_csv(index=False)
    for name,payload in [('README.md',final),('preparation.json',json.dumps(info,indent=2)),('split_manifest.csv',public_manifest)]:
        api.upload_file(path_or_fileobj=payload.encode(),path_in_repo=name,repo_id=repo_id,repo_type='dataset')
    revision=api.repo_info(repo_id,repo_type='dataset').sha
    print('Public dataset:',f'https://huggingface.co/datasets/{repo_id}')
    print('Pinned revision:',revision)
    return revision


def materialize_dataset(dataset,info,destination):
    """Turn downloaded embedded images into local paths, retaining exact prepared splits."""
    dest=Path(destination); images=dest/'images'; images.mkdir(parents=True,exist_ok=True); rows=[]
    for split in ['train','validation','test']:
        if split not in dataset: raise ValueError('Missing HF dataset split: '+split)
        ds=dataset[split].cast_column('image',HFImage(decode=False))
        for r in tqdm(ds,total=len(ds),desc=f'Preparing {split}',unit='image'):
            iid=r['image_id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+',iid): raise ValueError('Unsafe image ID')
            entry=r['image']; data=entry.get('bytes')
            if data is None:
                path=entry.get('path')
                if not path or not Path(path).is_file(): raise ValueError('Dataset does not contain embedded image bytes')
                data=Path(path).read_bytes()
            if hashlib.sha256(data).hexdigest()!=r['sha256']: raise ValueError('Image checksum mismatch: '+iid)
            path=images/(iid+'.jpg')
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']: path.write_bytes(data)
            rows.append(dict(image_id=iid,lesion_id=r['lesion_id'],target=r['target'],
                             labeled=r['labeled'],split=split,**{'class':r['class_name'],'path':str(path.resolve())}))
    frame=validate_manifest(pd.DataFrame(rows))
    if len(frame)!=info['num_images'] or info['classes']!=CLASSES: raise ValueError('Dataset does not match preparation.json')
    frame.to_csv(dest/'split_manifest.csv',index=False)
    (dest/'preparation.json').write_text(json.dumps(info,indent=2))
    return dest/'split_manifest.csv'
