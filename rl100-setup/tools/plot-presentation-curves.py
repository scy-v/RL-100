"""Two six-panel training figures; subordinate loss axes retain their own scales."""
import collections
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'rl100-setup/presentations/curves'
hist=json.loads((OUT/'training-histories.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':8,
    'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#AEBBC3','axes.labelcolor':'#34434F','xtick.color':'#53636E',
    'ytick.color':'#53636E','grid.color':'#E2E8ED','grid.linewidth':.6,
    'legend.frameon':False,'figure.facecolor':'white','savefig.facecolor':'white'})
BLUE,TEAL,ORANGE='#315B7A','#2E8173','#B4783D'
def epoch_average(rows,key='train_loss'):
    grouped=collections.defaultdict(list)
    for r in rows:
        if key in r and 'epoch' in r:grouped[int(r['epoch'])+1].append(r[key])
    x=sorted(grouped);return x,[np.mean(grouped[e]) for e in x]
def series(rows,key):return [r[key] for r in rows if key in r]
def tidy(ax):
    ax.grid(axis='y',alpha=.7);ax.tick_params(labelsize=7,length=3,pad=2)
def legend(ax,ncol=1):ax.legend(loc='best',fontsize=7,ncol=ncol,handlelength=1.8,columnspacing=.8)
manifest={}
for name,runs in hist.items():
    bc=runs[0]['rows'];online=runs[-1]['rows']
    selected_marker='20261020_145727' if 'Diffusion' in name else '20261020_152023'
    selected=next(r for r in runs if selected_marker in r['file'])
    offline=selected['rows'];ev=[r for r in online if 'online ppo success rates' in r]
    fig=plt.figure(figsize=(12.25,5.15))
    grid=fig.add_gridspec(2,3,left=.065,right=.993,bottom=.075,top=.92,wspace=.30,hspace=.56,height_ratios=[1,1.28])
    tops=[fig.add_subplot(grid[0,c]) for c in range(3)]
    for ax,title in zip(tops,['BC','Offline RL','Online PPO']):
        ax.set_title(title,pad=17,fontweight='bold',color='#25465F')
        ax.set_ylim(-3,105);ax.set_yticks([0,25,50,75,100]);tidy(ax)
    rows=[r for r in bc if 'test_mean_score' in r]
    tops[0].plot([r['epoch']+1 for r in rows],[r['test_mean_score']*100 for r in rows],'-o',color=BLUE,ms=3,lw=1.6,label='BC policy')
    tops[0].set(xlabel='Epoch',ylabel='Success (%)');legend(tops[0])
    other_runs=[r for r in runs if r is not selected and any('normal_eval_scores' in v for v in r['rows'])]
    for idx,other in enumerate(other_runs):
        yy=series(other['rows'],'normal_eval_scores')
        tops[1].plot(np.arange(len(yy))*1000,np.array(yy)*100,color='#BBC6CE',lw=.9,alpha=.75,
            label='Other configs' if idx==0 else None,zorder=1)
    for key,color,label in [('normal_eval_scores',BLUE,'Policy'),('ema_eval_scores',TEAL,'EMA')]:
        yy=series(offline,key)
        tops[1].plot(np.arange(len(yy))*1000,np.array(yy)*100,'-o',ms=3,lw=1.6,color=color,label=label)
    chosen_scores=series(offline,'normal_eval_scores')
    chosen_index=int(np.argmax(chosen_scores));chosen_x=chosen_index*1000;chosen_y=chosen_scores[chosen_index]*100
    tops[1].scatter([chosen_x],[chosen_y],marker='*',s=85,c=ORANGE,zorder=6,edgecolors='white',linewidths=.5)
    tops[1].annotate(f'Selected: {chosen_y:.1f}%',xy=(chosen_x,chosen_y),xytext=(chosen_x+250,chosen_y+17),
        color=ORANGE,fontsize=7,arrowprops={'arrowstyle':'-','color':ORANGE,'lw':.8})
    tops[1].set(xlabel='Policy updates')
    tops[1].legend(loc='lower left',fontsize=7,ncol=2,handlelength=1.8,columnspacing=.8)
    x=np.arange(len(ev))*.02
    for key,color,label in [('online ppo success rates',BLUE,'Policy'),('online ema success rates',TEAL,'EMA'),('cm_success rates',ORANGE,'Distilled')]:
        tops[2].plot(x,[r[key]*100 for r in ev],color=color,lw=1.6,label=label)
    tops[2].set(xlabel='Environment steps (million)',xlim=(0,1));legend(tops[2],3)
    # One BC loss panel; offline and online each contain three vertically stacked loss axes.
    ax=fig.add_subplot(grid[1,0]);xx,yy=epoch_average(bc)
    ax.plot(xx,yy,color=BLUE,lw=1.35);ax.set(xlabel='Epoch',ylabel='Loss',yscale='log');tidy(ax)
    ax.text(.03,.9,'BC loss',transform=ax.transAxes,color=BLUE,fontsize=8)
    sub=grid[1,1].subgridspec(3,1,hspace=.68)
    a=fig.add_subplot(sub[0]);q=series(bc,'Q_loss');v=series(bc,'value_loss')
    a.plot(np.arange(1,len(q)+1),q,color=BLUE,lw=1,label='Q')
    a.plot(np.arange(1,len(v)+1),v,color=TEAL,lw=1,label='V')
    a.set_yscale('log');a.set_ylabel('Critic',fontsize=8);tidy(a);legend(a,2)
    a.text(.99,1.04,'Epoch',transform=a.transAxes,ha='right',fontsize=6,color='#53636E')
    a=fig.add_subplot(sub[1])
    for key,col,label in [('loss/dynamics_train_loss',BLUE,'Train'),('loss/dynamics_holdout_loss',ORANGE,'Val')]:
        yy=series(bc,key);a.plot(np.arange(1,len(yy)+1),yy,color=col,lw=1,label=label)
    a.set_ylabel('Dynamics',fontsize=8);tidy(a);legend(a,2)
    a.text(.99,1.04,'Epoch',transform=a.transAxes,ha='right',fontsize=6,color='#53636E')
    a=fig.add_subplot(sub[2])
    for other in other_runs:
        zz=series(other['rows'],'dpg_loss')
        if zz:
            bins_other=[np.mean(zz[i:i+100]) for i in range(0,len(zz),100)]
            a.plot(np.arange(len(bins_other))*100+50,bins_other,color='#BBC6CE',lw=.7,alpha=.6)
    yy=series(offline,'dpg_loss')
    bins=[np.mean(yy[i:i+100]) for i in range(0,len(yy),100)]
    a.plot(np.arange(len(bins))*100+50,bins,color=BLUE,lw=1.1);a.set(xlabel='Policy updates',ylabel='Policy');tidy(a)
    sub=grid[1,2].subgridspec(3,1,hspace=.68)
    a=fig.add_subplot(sub[0]);xx,yy=epoch_average(online)
    a.plot(xx,yy,color=ORANGE,lw=1.1);a.set_yscale('log');a.set_ylabel('Pre-distill',fontsize=8);tidy(a)
    a.text(.99,1.04,'Epoch',transform=a.transAxes,ha='right',fontsize=6,color='#53636E')
    for row,key,col,label in [(1,'online actor_loss',BLUE,'PPO actor'),(2,'online critic_loss',TEAL,'PPO critic')]:
        a=fig.add_subplot(sub[row]);yy=series(online,key)
        a.plot(np.arange(1,len(yy)+1)*.02,yy,color=col,lw=1.1);a.set_ylabel(label,fontsize=8);tidy(a)
        if row==2:a.set_xlabel('Environment steps (million)');a.set_yscale('log')
    slug='3d-diffusion' if 'Diffusion' in name else '3d-flow-single'
    fig.savefig(OUT/(slug+'-curves.png'),dpi=240)
    fig.savefig(OUT/(slug+'-curves.pdf'))
    plt.close(fig)
    manifest[name]={'figure':slug+'-curves.png','selected_offline_run':selected['file'],
        'selected_offline_update':chosen_x,'selected_offline_success':chosen_y/100,
        'other_offline_configurations':len(other_runs),
        'bc_epochs':len(epoch_average(bc)[0]),'online_evaluations':len(ev),
        'final_online_success':{k:ev[-1][k] for k in ['online ppo success rates','online ema success rates','cm_success rates']},
        'loss_scope':'BC epoch means; offline selected configuration actor loss binned by 100 updates. Critic/dynamics separately pretrained; independent axes. Pre-distill epoch means; online PPO has 50 logged aggregate losses, ongoing distill loss not separately logged.'}
(OUT/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
