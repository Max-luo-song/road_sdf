from pathlib import Path
import json, csv, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/inspire/hdd/global_user/guoluosong-253108120129/chery_workspace/zhengzhehao/output/qcraft_20260613_102138_QCJPTDJ07384_3298_3308')
NEW = ROOT/'20261019_lidar+traincam2_3_5_9_12_0_1_6_7_10_11sdf-heightv1'
OLD = ROOT/'20260919_lidar+traincam2_3_5_9_12_0_1_6_7_10_11soft-road-bg-suppression'
DATA = Path('/inspire/hdd/global_user/guoluosong-253108120129/chery_workspace/zhengzhehao/processed/training/20260613_102138_QCJPTDJ07384_3298_3308')
OUT = Path('docs/experiments/20261019_road_height_v1')
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
def read(p): return np.asarray(Image.open(p).convert('RGB'))
def psnr(x): return float(-10*np.log10(max(float(x),1e-12)))
def analyze_image(p):
    frame, camera = map(int,p.stem.split('_'))
    gt=read(NEW/'videos/full_set_40000_gt_rgbs'/p.name)
    gt_old=read(OLD/'videos/full_set_40000_gt_rgbs'/p.name)
    new=read(p);old=read(OLD/'videos/full_set_40000_rgbs'/p.name)
    mask=np.asarray(Image.open(DATA/'road_masks'/f'{frame:06d}_{camera}.png'))>0
    ego=np.asarray(Image.open(DATA/'ego_masks'/f'{camera}.png'))>0
    valid=~ego;road=mask&valid
    # Static road excludes image-space dynamic labels when available.
    dynamic_file=DATA/'dynamic_masks'/f'{frame:06d}_{camera}.png'
    dynamic=np.asarray(Image.open(dynamic_file))>0 if dynamic_file.exists() else np.zeros_like(mask)
    static=road&~dynamic
    result=dict(frame=frame,camera=camera,gt_equal=bool(np.array_equal(gt,gt_old)),road_pixels=int(road.sum()))
    for mode,render in [('old',old),('new',new)]:
        error=((render.astype(np.float32)-gt.astype(np.float32))/255.)**2
        for label,m in [('full',np.ones_like(mask)),('valid',valid),('road',road),('static_road',static),('nonroad',valid&~mask)]:
            if m.any(): result[f'{mode}_{label}_psnr']=psnr(error[m].mean())
    result['road_delta']=result['new_road_psnr']-result['old_road_psnr']
    return result
paths=sorted((NEW/'videos/full_set_40000_rgbs').glob('*.png'))
if (OUT/'per_frame_metrics.csv').exists():
    rows=list(csv.DictReader((OUT/'per_frame_metrics.csv').open()))
    rows=[{k: (v == 'True' if k == 'gt_equal' else int(v) if k in ['frame','camera','road_pixels'] else float(v)) for k,v in row.items()} for row in rows]
else:
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(analyze_image,paths))
with (OUT/'per_frame_metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary={'images':len(rows),'gt_all_equal':all(r['gt_equal'] for r in rows),'cameras':{}}
for camera in sorted({r['camera'] for r in rows}):
    subset=[r for r in rows if r['camera']==camera]
    summary['cameras'][camera]={k:float(np.mean([r[k] for r in subset])) for k in rows[0] if k.endswith('psnr') or k=='road_delta'}
summary['means']={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k.endswith('psnr') or k=='road_delta'}
summary['road_win_fraction']=float(np.mean([r['road_delta']>0 for r in rows]))
summary['road_delta_percentiles']=dict(zip(['p0','p25','p50','p75','p100'],np.percentile([r['road_delta'] for r in rows],[0,25,50,75,100]).tolist()))
summary['official_group0']={}
for mode,path in [('old',OLD),('new',NEW)]:
    f=next((path/'group0/metrics_eval').glob('images_full*.json'))
    summary['official_group0'][mode]=json.loads(f.read_text())
    logs=[json.loads(l) for l in (path/'group0/metrics.json').read_text().splitlines() if l.strip()]
    summary.setdefault('training',{})[mode]={'records':len(logs),'mean_iter_time':float(np.mean([r['iter_time'] for r in logs[1:]])),'last':logs[-1]}
new_ck=torch.load(NEW/'group0/checkpoint_final.pth',weights_only=False,map_location='cpu',mmap=True)
old_ck=torch.load(OLD/'group0/checkpoint_final.pth',weights_only=False,map_location='cpu',mmap=True)
a=new_ck['models']['RoadNodes'];b=old_ck['models']['RoadNodes']
dz=a['_means'][:,2]-b['_means'][:,2]
summary['geometry']={'road_count':len(dz),'xy_identical':bool(torch.equal(a['_means'][:,:2],b['_means'][:,:2])),
    'delta_z_mean_m':float(dz.mean()),'delta_z_abs_mean_m':float(dz.abs().mean()),
    'delta_z_abs_percentiles_m':torch.quantile(dz.abs(),torch.tensor([0.,.5,.9,.95,.99,1.])).tolist(),
    'delta_z_gt_1cm_fraction':float((dz.abs()>.01).float().mean()),
    'delta_z_gt_5cm_fraction':float((dz.abs()>.05).float().mean()),
    'geometry_last_layer_weight_absmax':float(a['surface_field.geometry.2.weight'].abs().max()),
    'geometry_last_layer_bias_absmax':float(a['surface_field.geometry.2.bias'].abs().max())}
from models.road_surface import RoadSurfaceField
cfg=OmegaConf.load(NEW/'group0/config.yaml').model.RoadNodes.networks.road_surface
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(2718)
    initial=RoadSurfaceField(cfg)
params={k[len('surface_field.') :]:v for k,v in a.items() if k.startswith('surface_field.')}
summary['geometry']['all_network_parameters_match_initialization']=all(torch.equal(value,params[key]) for key,value in initial.named_parameters())
summary['geometry']['changed_network_parameters']=[key for key,value in initial.named_parameters() if not torch.equal(value,params[key])]
summary['gaussian_counts']={mode:{k:len(v['_means']) for k,v in ck['models'].items() if '_means' in v} for mode,ck in [('old',old_ck),('new',new_ck)]}
changed=[]
for folder in ['models','datasets','utils','tools']:
    for f in (NEW/'group0/backup'/folder).rglob('*.py'):
        relative=f.relative_to(NEW/'group0/backup');other=OLD/'group0/backup'/relative
        if not other.exists() or hashlib.sha256(f.read_bytes()).digest()!=hashlib.sha256(other.read_bytes()).digest(): changed.append(str(relative))
summary['changed_backup_python_files']=changed
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['training','official_group0']},indent=2),flush=True)

plt.rcParams.update({'font.size':10})
# Scientific comparison panels: identical frame and camera, no image enhancements.
cameras=[2,3,5,9,12]
fig,axes=plt.subplots(5,3,figsize=(18,15))
for i,cam in enumerate(cameras):
    for j,(label,path) in enumerate([('Ground truth',NEW/'videos/full_set_40000_gt_rgbs'),('Baseline',OLD/'videos/full_set_40000_rgbs'),('Height v1',NEW/'videos/full_set_40000_rgbs')]):
        axes[i,j].imshow(read(path/f'000050_{cam:03d}.png'));axes[i,j].set_title(f'{label} | camera {cam} | frame 50');axes[i,j].axis('off')
fig.tight_layout();fig.savefig(OUT/'original_frame50.png',dpi=120);plt.close(fig)
fig,axes=plt.subplots(5,2,figsize=(16,18))
for i,cam in enumerate(cameras):
    for j,(label,path) in enumerate([('Baseline',OLD),('Height v1',NEW)]):
        axes[i,j].imshow(read(path/f'novel_traj/left_shift_3m_step40000/images/000050/{cam}_rgbs.png'));axes[i,j].set_title(f'{label} | left 3m | camera {cam} | frame 50');axes[i,j].axis('off')
fig.tight_layout();fig.savefig(OUT/'novel_frame50.png',dpi=120);plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(16,10))
for i,frame in enumerate([0,50,98]):
    for j,(label,path) in enumerate([('Baseline',OLD),('Height v1',NEW)]):
        im=read(path/f'novel_traj/left_shift_3m_step40000/images/{frame:06d}/2_rgbs.png')
        axes[i,j].imshow(im[260:]);axes[i,j].set_title(f'{label} | left 3m | cam 2 | frame {frame} | road crop');axes[i,j].axis('off')
fig.tight_layout();fig.savefig(OUT/'novel_front_road.png',dpi=140);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4))
x=np.arange(5)
for offset,mode,label in [(-.17,'old','Baseline'),(.17,'new','Height v1')]:
    values=[summary['cameras'][c][mode+'_road_psnr'] for c in cameras]
    axes[0].bar(x+offset,values,.34,label=label)
axes[0].set_xticks(x,[str(c) for c in cameras]);axes[0].set_xlabel('Camera');axes[0].set_ylabel('Road-only PSNR (dB)');axes[0].legend()
axes[1].hist(dz.numpy()*100,bins=100,range=(-10,10));axes[1].set_xlabel('New Z - baseline Z (cm)');axes[1].set_ylabel('Road Gaussian count');axes[1].set_title('Geometry change: interpolation only (network unchanged)')
fig.tight_layout();fig.savefig(OUT/'metrics_and_height.png',dpi=150);plt.close(fig)
