import json,time,shutil
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd

class CloudBackup:
    def __init__(self,repo,run_id,token,reference):
        self.api=HfApi(token=token);self.repo=repo;self.token=token;self.prefix=run_id.strip('/')
        if not self.prefix or '..' in self.prefix:raise ValueError('Invalid run ID')
        self.reference=reference;self.index=None;self.display_handle=None
        self.api.create_repo(repo_id=repo,repo_type='model',private=True,exist_ok=True)
        self.head=self.api.repo_info(repo,repo_type='model').sha
        files=self.api.list_repo_files(repo,repo_type='model',revision=self.head)
        if self.prefix+'/latest.json' in files:
            self.index=json.loads(Path(hf_hub_download(repo,self.prefix+'/latest.json',revision=self.head,token=token)).read_text())
            if self.index['dataset']!=reference:raise ValueError('This run ID uses a different dataset revision; choose a new RUN_ID.')
        print('Checkpoint repository: https://huggingface.co/'+repo,flush=True)
    def restore(self,directory):
        if not self.index:
            print('No uploaded checkpoint for this run; starting fresh.',flush=True);return ''
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for remote,local in [(self.index['latest'],'last.pt'),(self.index.get('best'),'best.pt')]:
            if remote:
                src=hf_hub_download(self.repo,remote,revision=self.head,token=self.token)
                shutil.copy2(src,directory/local)
        (directory/'hf_dataset_reference.json').write_text(json.dumps(self.reference))
        print('Restored completed epoch',self.index['epoch'],flush=True)
        return str(directory/'last.pt')
    def table(self,history,status):
        from IPython.display import display,HTML
        import pandas as pd
        h=pd.DataFrame(history).copy()
        columns={'epoch':'Epoch','stage':'Stage','loss':'Loss','base':'Supervised','unsupervised':'Unsupervised',
                 'ncfr':'NCFR','graph':'Graph','val_accuracy':'Val accuracy %','val_macro_f1':'Val macro F1 %',
                 'pseudo_acceptance':'Pseudo-labels %','lr':'LR','seconds':'Seconds'}
        for k in ['val_accuracy','val_macro_f1','pseudo_acceptance']:h[k]*=100
        h=h[list(columns)].rename(columns=columns)
        html='<h4>'+status+'</h4><div style="max-height:480px;overflow:auto">'+h.to_html(index=False,float_format=lambda v:f'{v:.4f}',escape=True)+'</div>'
        if self.display_handle:self.display_handle.update(HTML(html))
        else:self.display_handle=display(HTML(html),display_id=True)
    def commit(self,files,message):
        for attempt in range(3):
            try:
                # Rebuild operations for every retry: Hub mutates them during upload.
                ops=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=payload) for name,payload in files]
                result=self.api.create_commit(repo_id=self.repo,repo_type='model',operations=ops,commit_message=message,parent_commit=self.head)
                self.head=result.oid;return
            except Exception:
                if attempt==2:raise RuntimeError('HF backup failed. Training stopped; local last.pt is retained. Fix the connection/token and resume; do not discard this session.') from None
                print(f'Backup retry {attempt+1}/2...',flush=True);time.sleep(3)
    def __call__(self,out,epoch,improved,history):
        out=Path(out);remote=f'{self.prefix}/checkpoints/epoch_{epoch:03d}.pt'
        best=remote if improved else (self.index or {}).get('best')
        index=dict(epoch=epoch,latest=remote,best=best,dataset=self.reference)
        self.table(history,f'Epoch {epoch}: uploading checkpoint to HF…')
        files=[(remote,str(out/'last.pt')),(self.prefix+'/latest.json',json.dumps(index).encode())]
        for name in ['history.csv','config.json','environment.json','hf_dataset_reference.json']:
            if (out/name).exists():files.append((self.prefix+'/'+name,str(out/name)))
        self.commit(files,f'{self.prefix}: completed epoch {epoch}')
        self.index=index
        self.table(history,f'Epoch {epoch}: checkpoint backed up to HF ✓')
        print(f'HF BACKUP CONFIRMED: epoch {epoch} | {remote}',flush=True)
    def final_outputs(self,out):
        out=Path(out)
        files=[(self.prefix+'/results/'+p.name,str(p)) for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.png']]
        self.commit(files,self.prefix+': final evaluation artifacts')
        print('Final metrics and plots backed up to HF.',flush=True)
