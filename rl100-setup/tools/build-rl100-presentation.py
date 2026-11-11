"""Build matching native-table and raster-slide academic PowerPoint decks."""
import collections.abc
import json
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/rl100-ppt-deps')
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'rl100-setup/presentations'
OUT.mkdir(parents=True, exist_ok=True)
FONT_PATH = Path('/tmp/rl100-ppt-font.ttf')
FONT_NAME = 'Noto Sans SC'
W, H, SCALE = 13.333333, 7.5, 180
NAVY, INK, GRAY = '25465F', '202C36', '667784'
LINE, PALE, TEAL, AMBER = 'D5DEE4', 'F2F5F7', '27756C', '986B2B'

slides = [
 dict(title='三类训练路线', label='METHOD OVERVIEW', kind='overview',
      rows=[['RL-100', 'BC → 离线策略迭代 → 蒸馏 → 在线 PPO'],
            ['evo-rl  (Recap)', '迭代式离线 RL'],
            ['RLinf  (RLtoken)', '在线 RL']],
      foot='本次汇报：方法路线与 RL-100 实验',
      notes='Recap 与 RLtoken 的分类沿用用户给定的汇报框架；此页不扩展其具体算法。RL-100 的离线阶段用“离线策略迭代”，避免误标为普通在线 PPO。'),
 dict(title='配置组合', label='RL-100  /  01',
      headers=['配置维度', '可选项'], widths=[3.4,8.6],
      rows=[['观测类型','3D 点云 / 2D 图像'],['策略方法','Diffusion / Flow'],
            ['动作形式','单步动作 / 多步动作块'],['动作网络','SkipNet / U-Net']],
      y=1.75, heights=[.63,.78,.78,.78,.78], font=21,
      callout='独立组合：16 种     常用配对：8 种',
      foot='常用配对：单步 + SkipNet；动作块 + U-Net',
      notes='这里展示配置空间，不表示作者报告了全部组合的实验结果。'),
 dict(title='五组训练实验', label='RL-100  /  02',
      headers=['实验','观测','动作','网络','状态'], widths=[3.15,1.25,1.3,1.65,4.65],
      rows=[['3D Diffusion','点云','单步','SkipNet','已完成；最终成功率 100%'],
            ['2D Diffusion','图像','单步','SkipNet','在线 PPO 中'],
            ['3D Flow 单步','点云','单步','SkipNet','已完成；最终成功率 0%'],
            ['3D Flow chunk','点云','16 步','U-Net','离线策略优化中'],
            ['2D Flow chunk','图像','16 步','U-Net','BC 中断；602 / 800 epoch']],
      y=1.7, heights=[.65,.75,.75,.75,.75,.75], font=16,
      foot='训练中实验：沿用上次状态快照；已完成实验：最终成功率',
      notes='训练中实验状态并非本次导出时重新实时扫描。本次重新提取两个已完成实验的完整曲线：均有 51 个在线评估点（初始 + 每 2 万步），最终 100 万步，W&B exit 0；3D Flow 单步主策略、EMA、CM 最终均为 0；3D Diffusion 三者最终均为 100%。'),
 dict(title='阶段耗时 · 离线', label='RL-100  /  03A',
      headers=['训练阶段','3D\nDiffusion','2D\nDiffusion','3D Flow\n单步','3D Flow\nchunk','2D Flow\nchunk'],
      widths=[2.05,1.99,1.99,1.99,1.99,1.99],
      rows=[['BC','1h18m (1)','1h45m (4)','1h43m (1)','3h22m (4)','≈7h33m (2)'],
            ['Critic','44m (1)','1h12m (4)','54m (1)','33m (4)','—'],
            ['Dynamics','19m (1)','8m (4)','30m (1)','1h25m (4)','—'],
            ['策略搜索','3h59m (1)','7h34m (≤4)','≈3h18m (1)','≈5.5–6h (4)','—'],
            ['离线合计','≈6h20m','≈10h39m','≈6h25m','≈11–12h','待定']],
      y=1.75, heights=[.78,.73,.73,.73,.73,.75], font=15,
      foot='括号：GPU 卡数；≈：约数或预估。2D Flow chunk 的 BC 已运行 5h41m 后中断。',
      notes='3D Diffusion 离线搜索 8 组完成；2D Diffusion 10 组完成、最多 4 卡并行；3D Flow 单步 5 组完成、第 6 组中断，耗时包括实际已运行的部分。3D Flow chunk 搜索为 4 组并行，每组单卡；预计耗时为此前约 500 次更新时外推，未完成，不是实测终值。离线合计不含在线前蒸馏。来源：reports/offline-stage-timings.json 与 /tmp/rl100_flow_single_status.json、此前 Flow 状态日志。'),
 dict(title='阶段耗时 · 在线与总计', label='RL-100  /  03B',
      headers=['训练阶段','3D\nDiffusion','2D\nDiffusion','3D Flow\n单步','3D Flow\nchunk','2D Flow\nchunk'],
      widths=[2.05,1.99,1.99,1.99,1.99,1.99],
      rows=[['在线前蒸馏','21m (1)','1h57m (1)','23m (1)','—','—'],
            ['在线 PPO\n100 万步','≈7h40m (1)','≈53h (1)','≈4h49m (1)','—','—'],
            ['全流程总计','≈14h20m','≈66h','≈11h40m','待定','待定']],
      y=2.05, heights=[.8,.9,.95,.95], font=15,
      callout='已完成：3D Diffusion、3D Flow 单步',
      foot='括号：GPU 卡数；2D Diffusion 为此前日志速率外推；在线耗时包含持续蒸馏。',
      notes='所有估计均沿用对话中已确认的历史日志快照，未当作最新完成时间。Flow chunk 的 11–12h 仅为离线流程，不能与其他实验的全流程耗时直接比较。在线为单卡。3D Flow 单步训练完成但策略崩溃，耗时较短不能视为更快收敛。'),
 dict(title='2D / 3D · 更新速度', label='RL-100  /  04',
      headers=['策略配置','2D\nms / 更新','3D\nms / 更新','2D / 3D\n耗时比','额外耗时中\n编码占比'],
      widths=[4.4,1.85,1.85,1.85,2.05],
      rows=[['Diffusion + SkipNet · 单步','199.63','31.83','6.27×','97.7%'],
            ['Flow + SkipNet · 单步','200.66','35.22','5.70×','97.0%'],
            ['Diffusion + U-Net · 16 步','236.82','72.19','3.28×','97.5%'],
            ['Flow + U-Net · 16 步','245.22','74.06','3.31×','96.3%']],
      y=1.7, heights=[.85,.82,.82,.82,.82], font=16,
      callout='主要差异：观测编码',
      foot='单卡 RTX 4090 D · FP32 · batch 128 · BC 更新；来源：summary',
      notes='逐字采用 rl100-setup/summary.md 的原始测速表。预热 5 步，测量 20 步中位数，3 帧观测，CUDA_LAUNCH_BLOCKING=1，包含 H2D/前向/反向/优化器/EMA。Diffusion+SkipNet 为 GPU4 旧记录，其他为 GPU2；每行 2D 与 3D 在同一 GPU 测量，保留各自配方。峰值分配显存（MiB），前两行 2D/3D 为 9416/2097，后两行为 10826/3428。峰值显存不放主表以减少密度。'),
 dict(title='网络与动作长度 · 控制变量测速', label='RL-100  /  05',
      headers=['组合','U-Net / SkipNet\n同为 4 步','U-Net / SkipNet\n同为 16 步','SkipNet\n16 步 / 单步','U-Net\n16 步 / 4 步'],
      widths=[3.0,2.25,2.25,2.25,2.25],
      rows=[], y=1.7, heights=[.85,.8,.8,.8,.8], font=16,
      callout='换网络影响大；增加动作长度影响小',
      foot='耗时比 · BC 更新 · 单卡 / FP32 / batch 128；原版 U-Net 不支持长度 1。',
      notes='原始文件 reports/network-chunk-speed.json；3 帧观测，Mish，固定各模态基础配方，测试 2 模态×2 方法×2 网络×3 长度，共 24 个配置，20 个可运行，4 个 U-Net 长度 1 原生维度报错。每项预热 4 次、3 轮每轮 6 次取中位数。部分早期 2D 项波动，经复测替换，首轮记录保存在 first_pass_rows。仅动作网络更新（前向/反向/优化器/EMA）时 U-Net/SkipNet 为约 9.7–11.1 倍。0.94 等小于 1 的短测结果不说明 chunk 有固有加速。'),
]
benchmark = json.loads((ROOT / 'rl100-setup/reports/network-chunk-speed.json').read_text())
for r in benchmark['comparisons']:
    slides[-1]['rows'].append([r['modality'].upper() + ' ' + r['family'].title()] +
        [f"{r[k]:.2f}×" for k in ['unet_skip_4','unet_skip_16','skip_16_1','unet_16_4']])

for label,slug in [('3D Diffusion','3d-diffusion'),('3D Flow 单步','3d-flow-single')]:
    figure=OUT/'curves'/(slug+'-curves.png')
    if figure.exists():
        slides.append(dict(title=label+' · Success / Loss',label='RL-100  /  TRAINING',kind='curves',
            figure=str(figure),foot='BC → Offline iteration → Online · Ongoing distillation loss was not logged separately.',
            notes='两行三列：上排成功率，下排 loss；三列分别为 BC、离线迭代、在线。离线列的 loss 内部再分 Critic、Dynamics、所选配置的策略优化。在线列的 loss 内部再分在线前蒸馏、PPO actor、PPO critic；在线持续蒸馏 loss 没有单独记录，不编造曲线。BC loss 为每 epoch 均值；离线策略 loss 为每 100 次更新均值。各 loss 独立纵轴，Critic、BC、预蒸馏和在线 critic 使用对数刻度。数据来源和最终成功率见 curves/figure-manifest.json。'))

video_meta=OUT/'video/selected-video.json'
if video_meta.exists():
    import imageio
    meta=json.loads(video_meta.read_text())
    reader=imageio.get_reader(meta['video']);frame=reader.get_data(75);reader.close()
    poster=OUT/'video/poster.png';Image.fromarray(frame).save(poster)
    slides.append(dict(title='3D Diffusion · Adroit Door',label='RL-100  /  SIMULATION',kind='video',
        movie=meta['video'],poster=str(poster),
        foot='Online EMA · DDIM · seed 0 · 1920 × 1080 · 30 fps · Success',
        notes='可编辑版内嵌真实仿真视频，点击播放；图片版此页为静态海报，视频另附。在线 best EMA 检查点，确定性 DDIM 推理，seed 0，100 个控制步，goal_achieved 累计 67，超过 success_threshold=25。1080p 直接仿真渲染，不是低分辨率放大。10Hz 控制画面每帧重复三次编码到 30fps，以保持 10.1 秒展示节奏。'))

def all_strings(value):
    if isinstance(value,str): yield value
    elif isinstance(value,dict):
        for v in value.values(): yield from all_strings(v)
    elif isinstance(value,list):
        for v in value: yield from all_strings(v)

if '--font-text' in sys.argv:
    # Google Fonts text subsets keep the standalone font download small.
    Path('/tmp/rl100-ppt-font-text.txt').write_text(''.join(sorted(set(''.join(all_strings(slides))))) + '0123456789/—≈≤×→· ()%:')
    sys.exit(0)

def rgb(v): return RGBColor.from_string(v)
def px(v): return round(v*SCALE)
def font(size): return ImageFont.truetype(str(FONT_PATH), round(size*SCALE/72))
def wrap(text, size, width):
    d = ImageDraw.Draw(Image.new('RGB',(1,1)))
    result=[]
    for para in text.split('\n'):
        line=''
        for ch in para:
            if line and d.textlength(line+ch,font=font(size)) > px(width):
                result.append(line);line=ch
            else: line+=ch
        result.append(line)
    return result

def paint_text(draw, text, x,y,w,h,size=18,color=INK,align='left'):
    lines=wrap(text,size,w)
    advance=size/72*1.27
    assert advance*len(lines) <= h+.045, ('text overflow',text,lines,h)
    top=y+(h-advance*len(lines))/2
    for i,line in enumerate(lines):
        tw=draw.textlength(line,font=font(size))/SCALE
        xx=x if align=='left' else x+(w-tw)/2
        draw.text((px(xx),px(top+i*advance)),line,font=font(size),fill='#'+color,anchor='lt')

def textbox(slide,text,x,y,w,h,size=18,color=INK,bold=False,align='left'):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=box.text_frame;tf.clear();tf.word_wrap=True
    tf.margin_left=tf.margin_right=0;tf.margin_top=tf.margin_bottom=0
    tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    for j,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if j==0 else tf.add_paragraph()
        p.alignment=PP_ALIGN.LEFT if align=='left' else PP_ALIGN.CENTER
        p.space_before=p.space_after=Pt(0);p.line_spacing=1.15
        run=p.add_run();run.text=line
        run.font.name=FONT_NAME;run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=rgb(color)
        for tag in ['a:ea','a:cs']:
            e=OxmlElement(tag);e.set('typeface',FONT_NAME);run._r.get_or_add_rPr().append(e)
    return box

def rect(slide,x,y,w,h,color):
    shape=slide.shapes.add_shape(1,Inches(x),Inches(y),Inches(w),Inches(h))
    shape.fill.solid();shape.fill.fore_color.rgb=rgb(color);shape.line.fill.background()

native=Presentation();native.slide_width=Inches(W);native.slide_height=Inches(H)
native.core_properties.title='RL-100 实验与速度对比'
native.core_properties.subject='三类训练路线；五组实验；阶段耗时；控制变量测速'
native.core_properties.author='RL-100 实验记录'
pngs=[]
for i,spec in enumerate(slides,1):
    s=native.slides.add_slide(native.slide_layouts[6])
    im=Image.new('RGB',(px(W),px(H)),'white');d=ImageDraw.Draw(im)
    rect(s,.65,.55,.055,.92,NAVY);d.rectangle([px(.65),px(.55),px(.705),px(1.47)],fill='#'+NAVY)
    textbox(s,spec['label'],.87,.53,11.8,.3,11,GRAY)
    paint_text(d,spec['label'],.87,.53,11.8,.3,11,GRAY)
    textbox(s,spec['title'],.85,.91,11.8,.55,28,INK,True)
    paint_text(d,spec['title'],.85,.91,11.8,.55,28)
    if spec.get('kind')=='overview':
        for j,(name,method) in enumerate(spec['rows']):
            y=2.05+j*1.27
            textbox(s,name,.9,y,3.7,.65,25,NAVY,True)
            paint_text(d,name,.9,y,3.7,.65,25,NAVY)
            textbox(s,'→',4.5,y,.6,.65,28,TEAL)
            paint_text(d,'→',4.5,y,.6,.65,28,TEAL)
            textbox(s,method,5.25,y,7.2,.65,21)
            paint_text(d,method,5.25,y,7.2,.65,21)
            if j<2:
                rect(s,.9,y+.94,11.55,.008,LINE)
                d.line([px(.9),px(y+.94),px(12.45),px(y+.94)],fill='#'+LINE,width=1)
    elif spec.get('kind') in ['curves','video']:
        if spec['kind']=='curves':
            x,y,w,h=.67,1.64,12.0,5.045
            path=spec['figure']
            s.shapes.add_picture(path,Inches(x),Inches(y),width=Inches(w),height=Inches(h))
        else:
            x,y,w,h=2.42,1.75,8.5,4.78125
            path=spec['poster']
            s.shapes.add_movie(spec['movie'],Inches(x),Inches(y),Inches(w),Inches(h),
                poster_frame_image=path,mime_type='video/mp4')
        panel=Image.open(path).convert('RGB').resize((px(w),px(h)),Image.Resampling.LANCZOS)
        im.paste(panel,(px(x),px(y)))
    else:
        rows=[spec['headers']]+spec['rows'];x=.67;y=spec['y'];widths=spec['widths'];heights=spec['heights']
        table=s.shapes.add_table(len(rows),len(widths),Inches(x),Inches(y),Inches(sum(widths)),Inches(sum(heights))).table
        table.first_row=True
        for c,w in enumerate(widths):table.columns[c].width=Inches(w)
        for r,values in enumerate(rows):
            table.rows[r].height=Inches(heights[r]);xx=x
            for c,text in enumerate(values):
                cell=table.cell(r,c);cell.text=text
                bg=NAVY if r==0 else PALE if r%2 else 'FFFFFF'
                fg='FFFFFF' if r==0 else TEAL if c>0 and ('×' in text) else INK
                if r>0 and '≈' in text and ((i==4 and c>=4) or (i==5 and c==2)):fg=AMBER
                cell.fill.solid();cell.fill.fore_color.rgb=rgb(bg)
                cell.margin_left=Inches(.14);cell.margin_right=Inches(.12)
                cell.margin_top=cell.margin_bottom=Inches(.045)
                cell.vertical_anchor=MSO_ANCHOR.MIDDLE
                tcpr=cell._tc.get_or_add_tcPr()
                for tag in ['a:lnL','a:lnR','a:lnT','a:lnB']:
                    edge=OxmlElement(tag);edge.set('w','0');edge.append(OxmlElement('a:noFill'));tcpr.append(edge)
                for p in cell.text_frame.paragraphs:
                    p.alignment=PP_ALIGN.LEFT if c==0 else PP_ALIGN.CENTER
                    p.space_before=p.space_after=Pt(0);p.line_spacing=1.15
                    for run in p.runs:
                        run.font.name=FONT_NAME;run.font.size=Pt(spec['font']);run.font.color.rgb=rgb(fg);run.font.bold=(r==0)
                        ea=OxmlElement('a:ea');ea.set('typeface',FONT_NAME);run._r.get_or_add_rPr().append(ea)
                d.rectangle([px(xx),px(y),px(xx+widths[c]),px(y+heights[r])],fill='#'+bg)
                paint_text(d,text,xx+.14,y+.045,widths[c]-.26,heights[r]-.09,spec['font'],fg,'left' if c==0 else 'center')
                xx+=widths[c]
            y+=heights[r]
        if spec.get('callout'):
            textbox(s,spec['callout'],.85,6.18,11.8,.4,18,TEAL)
            paint_text(d,spec['callout'],.85,6.18,11.8,.4,18,TEAL)
    textbox(s,spec['foot'],.85,6.91,11.4,.3,10,GRAY)
    paint_text(d,spec['foot'],.85,6.91,11.4,.3,10,GRAY)
    textbox(s,f'{i:02}',12.2,6.91,.4,.3,10,GRAY)
    paint_text(d,f'{i:02}',12.2,6.91,.4,.3,10,GRAY)
    s.notes_slide.notes_text_frame.text=spec.get('notes','')
    png=OUT/f'slide-{i:02}.png';im.save(png);pngs.append(png)
native.save(OUT/'RL100_可编辑版.pptx')

raster=Presentation();raster.slide_width=Inches(W);raster.slide_height=Inches(H)
raster.core_properties.title='RL-100 实验与速度对比（图片版）'
for spec,png in zip(slides,pngs):
    s=raster.slides.add_slide(raster.slide_layouts[6])
    s.shapes.add_picture(str(png),0,0,width=Inches(W),height=Inches(H))
    s.notes_slide.notes_text_frame.text=spec.get('notes','')
raster.save(OUT/'RL100_图片版.pptx')

sheet=Image.new('RGB',(1600,((len(pngs)+1)//2)*490),'#E5E9EC')
for i,png in enumerate(pngs):
    im=Image.open(png);im.thumbnail((770,434))
    xx=15+(i%2)*800;yy=15+(i//2)*490
    sheet.paste(im,(xx,yy))
sheet.save(OUT/'预览总览.png')
(OUT/'slides-content.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2)+'\n')
print('Built',len(slides),'slides in each deck:',OUT)
