"""Compact academic decks: native tables vs table-only rasterization; all captions editable."""
import collections.abc
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, '/tmp/rl100-ppt-deps')
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'rl100-setup/presentations'
ASSETS=OUT/'compact-assets';ASSETS.mkdir(exist_ok=True)
original=json.loads((OUT/'presentation-data.json').read_text())
FONT='/tmp/rl100-ppt-font.ttf';TYPEFACE='Noto Sans SC'
W,H,SCALE=13.333333,7.5,200
NAVY,INK,GRAY='25465F','243340','6C7B86'
TEAL,AMBER,RED='2C766E','956B35','965950'
RULE,HEAD,STRIPE='C4D0D8','E9EFF3','F7F9FA'

combo=copy.deepcopy(original[1]);combo.update(widths=[1.12,3.0],font=14,
    heights=[.48,.59,.59,.59,.59])
experiments=copy.deepcopy(original[2]);experiments.update(
    widths=[2.12,.72,.68,1.05,2.78],font=13.5,heights=[.48]+[.59]*5,
    rows=[['3D Diffusion','点云','单步','SkipNet','完成 · 100%'],
          ['2D Diffusion','图像','单步','SkipNet','在线 PPO 中'],
          ['3D Flow 单步','点云','单步','SkipNet','完成 · 0%'],
          ['3D Flow chunk','点云','16 步','U-Net','离线策略优化中'],
          ['2D Flow chunk','图像','16 步','U-Net','BC 中断 · 602/800']])
offline=copy.deepcopy(original[3]);offline.update(font=14,heights=[.53]+[.39]*5)
online=copy.deepcopy(original[4]);online.update(font=14,heights=[.53,.41,.52,.41])
speed=copy.deepcopy(original[5]);speed.update(font=14,heights=[.55]+[.405]*4)
factorial=copy.deepcopy(original[6]);factorial.update(font=14,heights=[.55]+[.405]*4)

pages=[
 dict(kind='overview',title='三类训练路线',section='METHOD OVERVIEW',rows=original[0]['rows'],
      foot='本次汇报：方法路线与 RL-100 实验',notes=original[0]['notes']),
 dict(kind='tables',title='配置与实验',section='RL-100 / EXPERIMENTS',
      blocks=[dict(data=combo,x=.65,y=1.92,title='配置组合',title_y=1.42),
              dict(data=experiments,x=5.35,y=1.92,title='五组训练实验',title_y=1.42)],
      annotations=[('独立组合：16 种',.73,5.12,4.0,.35,15,TEAL),
                   ('常用配对：8 种',.73,5.53,4.0,.35,15,TEAL),
                   ('单步 + SkipNet；动作块 + U-Net',5.43,5.72,7.0,.35,14,GRAY)],
      foot='训练中实验：沿用上次状态快照；已完成实验：最终成功率',
      notes=combo['notes']+'\n'+experiments['notes']),
 dict(kind='tables',title='阶段耗时',section='RL-100 / TRAINING TIME',
      blocks=[dict(data=offline,x=.65,y=1.85,title='离线',title_y=1.46),
              dict(data=online,x=.65,y=4.81,title='在线与总计',title_y=4.41)],
      foot='括号：GPU 卡数；≈：约数或预估。2D Flow chunk 的 BC 已运行 5h41m 后中断。',
      notes=offline['notes']+'\n'+online['notes']),
 dict(kind='tables',title='更新速度对比',section='RL-100 / UPDATE SPEED',
      blocks=[dict(data=speed,x=.65,y=1.85,title='2D / 3D',title_y=1.46),
              dict(data=factorial,x=.65,y=4.61,title='网络与动作长度',title_y=4.21)],
      foot='BC 更新 · RTX 4090 D · FP32 · batch 128 · 原版 U-Net 不支持长度 1。',
      notes=speed['notes']+'\n'+factorial['notes']),
]
for label,slug in [('3D Diffusion','3d-diffusion'),('3D Flow 单步','3d-flow-single')]:
    pages.append(dict(kind='curves',title=label+' · Success / Loss',section='RL-100 / TRAINING CURVES',
        figure=str(OUT/'curves'/(slug+'-curves.png')),
        foot='BC → Offline RL → Online PPO · Gray: other configs · Selected checkpoint marked · Online distill loss not logged.',
        notes='两行三列，上排成功率，下排 loss；三列 BC、离线 RL、在线 PPO。浅灰曲线为其他超参数配置（含中断组已记录部分），蓝线为最终选中的配置，星号为实际采用的最佳检查点。单组 5000 次更新；各组从 BC 检查点重新开始，不能首尾相连。离线 loss 内含 Critic、Dynamics、策略；在线 loss 内含预蒸馏、PPO actor、PPO critic。不同 loss 独立纵轴，不对 loss 值做归一化。在线持续蒸馏没有独立 loss 记录。'))
meta=json.loads((OUT/'video/selected-video.json').read_text())
pages.append(dict(kind='video',title='3D Diffusion · Adroit Door',section='RL-100 / SIMULATION',
    movie=meta['video'],poster=str(OUT/'video/poster.png'),
    foot='Online EMA · DDIM · seed 0 · 1920 × 1080 · 30 fps · Success',
    notes='两个版本均保留内嵌 MP4 视频和可编辑标题。真实推理回合：100 个控制步，goal_achieved=67，大于阈值 25；1920×1080 直接渲染，30fps 编码。'))

def px(value):return round(value*SCALE)
def rgb(color):return RGBColor.from_string(color)
def font(size):return ImageFont.truetype(FONT,round(size*SCALE/72))
def wrap(text,size,width):
    draw=ImageDraw.Draw(Image.new('RGB',(1,1)));lines=[]
    for para in str(text).split('\n'):
        line=''
        for ch in para:
            if line and draw.textlength(line+ch,font=font(size))>px(width):lines.append(line);line=ch
            else:line+=ch
        lines.append(line)
    return lines

def draw_text(draw,text,x,y,w,h,size,color=INK,align='left'):
    lines=wrap(text,size,w);line_h=size/72*1.22
    assert len(lines)*line_h<=h+.02,('overflow',text,lines,w,h,size)
    top=y+(h-len(lines)*line_h)/2
    for i,line in enumerate(lines):
        width=draw.textlength(line,font=font(size))/SCALE
        left=x if align=='left' else x+(w-width)/2
        draw.text((px(left),px(top+i*line_h)),line,font=font(size),fill='#'+color,anchor='lt')

def native_text(slide,text,x,y,w,h,size,color=INK,bold=False,align='left'):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=box.text_frame;tf.clear();tf.word_wrap=True
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
    tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    for i,line in enumerate(str(text).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.line_spacing=1.12
        p.space_before=p.space_after=Pt(0);p.alignment=PP_ALIGN.LEFT if align=='left' else PP_ALIGN.CENTER
        run=p.add_run();run.text=line;run.font.name=TYPEFACE;run.font.size=Pt(size)
        run.font.color.rgb=rgb(color);run.font.bold=bold
        for name in ['a:ea','a:cs']:
            ea=OxmlElement(name);ea.set('typeface',TYPEFACE);run._r.get_or_add_rPr().append(ea)

def native_rect(slide,x,y,w,h,color):
    shape=slide.shapes.add_shape(1,Inches(x),Inches(y),Inches(w),Inches(h))
    shape.fill.solid();shape.fill.fore_color.rgb=rgb(color);shape.line.fill.background()

def table_style(data,row,col,text):
    bg=HEAD if row==0 else STRIPE if row%2 else 'FFFFFF'
    fg=NAVY if row==0 else INK
    bold=row==0
    if row and col and '×' in text:fg=TEAL
    if row and '完成 · 100%' in text:fg=TEAL
    if row and ('完成 · 0%' in text or '中断' in text):fg=RED
    if row and ('合计' in data['rows'][row-1][0] or '总计' in data['rows'][row-1][0]):bg='EBF3F1';bold=True
    if row and text.startswith('≈') and ((data['title'].endswith('离线') and col>=4) or ('在线与总计' in data['title'] and col==2)):fg=AMBER
    return bg,fg,bold

def table_image(data,path):
    widths=data['widths'];heights=data['heights'];rows=[data['headers']]+data['rows']
    im=Image.new('RGB',(px(sum(widths)),px(sum(heights))),'white');d=ImageDraw.Draw(im);y=0
    for r,values in enumerate(rows):
        x=0
        for c,text in enumerate(values):
            bg,fg,bold=table_style(data,r,c,text)
            d.rectangle([px(x),px(y),px(x+widths[c]),px(y+heights[r])],fill='#'+bg)
            size=12.5 if r==0 else data['font']
            draw_text(d,text,x+.10,y+.025,widths[c]-.20,heights[r]-.05,size,fg,'left' if c==0 else 'center')
            x+=widths[c]
        y+=heights[r]
        d.line([(0,px(y)-1),(im.width,px(y)-1)],fill='#'+RULE,width=1)
    d.line([(0,1),(im.width,1)],fill='#'+NAVY,width=2)
    d.line([(0,im.height-2),(im.width,im.height-2)],fill='#'+NAVY,width=2)
    im.save(path);return im

def native_table(slide,data,x,y):
    rows=[data['headers']]+data['rows'];widths=data['widths'];heights=data['heights']
    table=slide.shapes.add_table(len(rows),len(widths),Inches(x),Inches(y),Inches(sum(widths)),Inches(sum(heights))).table
    table.first_row=True
    for c,width in enumerate(widths):table.columns[c].width=Inches(width)
    for r,values in enumerate(rows):
        table.rows[r].height=Inches(heights[r])
        for c,text in enumerate(values):
            cell=table.cell(r,c);cell.text=text
            bg,fg,bold=table_style(data,r,c,text)
            cell.fill.solid();cell.fill.fore_color.rgb=rgb(bg)
            cell.margin_left=cell.margin_right=Inches(.10)
            cell.margin_top=cell.margin_bottom=Inches(.025);cell.vertical_anchor=MSO_ANCHOR.MIDDLE
            tcpr=cell._tc.get_or_add_tcPr()
            for edge in ['a:lnL','a:lnR','a:lnT','a:lnB']:
                element=OxmlElement(edge)
                visible=(edge=='a:lnB' or (edge=='a:lnT' and r==0))
                if visible:
                    heavy=(r==len(rows)-1 and edge=='a:lnB') or (r==0 and edge=='a:lnT')
                    element.set('w','10000' if heavy else '4000')
                    fill=OxmlElement('a:solidFill');color=OxmlElement('a:srgbClr');color.set('val',NAVY if heavy else RULE);fill.append(color);element.append(fill)
                else:element.append(OxmlElement('a:noFill'))
                tcpr.append(element)
            for p in cell.text_frame.paragraphs:
                p.alignment=PP_ALIGN.LEFT if c==0 else PP_ALIGN.CENTER
                p.space_before=p.space_after=Pt(0);p.line_spacing=1.12
                for run in p.runs:
                    run.font.name=TYPEFACE;run.font.size=Pt(12.5 if r==0 else data['font'])
                    run.font.color.rgb=rgb(fg);run.font.bold=bold
                    ea=OxmlElement('a:ea');ea.set('typeface',TYPEFACE);run._r.get_or_add_rPr().append(ea)

decks=[]
for title in ['RL-100 实验与速度对比','RL-100 实验与速度对比（表格图片版）']:
    p=Presentation();p.slide_width=Inches(W);p.slide_height=Inches(H)
    p.core_properties.title=title;p.core_properties.author='RL-100 实验记录';decks.append(p)
previews=[]
for number,page in enumerate(pages,1):
    slides=[p.slides.add_slide(p.slide_layouts[6]) for p in decks]
    im=Image.new('RGB',(px(W),px(H)),'white');d=ImageDraw.Draw(im)
    def text(value,x,y,w,h,size,color=INK,bold=False,align='left'):
        for s in slides:native_text(s,value,x,y,w,h,size,color,bold,align)
        draw_text(d,value,x,y,w,h,size,color,align)
    def line(x,y,w,color=RULE):
        for s in slides:native_rect(s,x,y,w,.008,color)
        d.line([(px(x),px(y)),(px(x+w),px(y))],fill='#'+color,width=1)
    text(page['section'],.68,.39,10.9,.25,10,GRAY)
    text(page['title'],.65,.78,12,.55,27,NAVY,True)
    line(.65,1.39,12.05)
    if page['kind']=='overview':
        for row,(name,method) in enumerate(page['rows']):
            y=2.02+row*1.23
            text(name,.85,y,3.5,.62,23,NAVY,True)
            text('→',4.37,y,.6,.62,26,TEAL)
            text(method,5.1,y,7.3,.62,20)
            if row<2:line(.85,y+.93,11.65)
    elif page['kind']=='tables':
        for index,block in enumerate(page['blocks']):
            data=block['data'];x,y=block['x'],block['y'];width=sum(data['widths']);height=sum(data['heights'])
            text(block['title'],x+.02,block['title_y'],width,.32,16,NAVY,True)
            asset=ASSETS/f'table-{number:02}-{index+1}.png';table_im=table_image(data,asset)
            native_table(slides[0],data,x,y)
            slides[1].shapes.add_picture(str(asset),Inches(x),Inches(y),width=Inches(width),height=Inches(height))
            im.paste(table_im,(px(x),px(y)))
        for args in page.get('annotations',[]):text(*args)
    elif page['kind']=='curves':
        x,y,width,height=.60,1.64,12.05,5.065
        for s in slides:s.shapes.add_picture(page['figure'],Inches(x),Inches(y),width=Inches(width),height=Inches(height))
        panel=Image.open(page['figure']).convert('RGB').resize((px(width),px(height)),Image.Resampling.LANCZOS)
        im.paste(panel,(px(x),px(y)))
    elif page['kind']=='video':
        x,y,width,height=2.30,1.68,8.72,4.905
        for s in slides:s.shapes.add_movie(page['movie'],Inches(x),Inches(y),Inches(width),Inches(height),poster_frame_image=page['poster'],mime_type='video/mp4')
        panel=Image.open(page['poster']).convert('RGB').resize((px(width),px(height)),Image.Resampling.LANCZOS);im.paste(panel,(px(x),px(y)))
    text(page['foot'],.68,6.98,11.7,.23,9,GRAY)
    text(f'{number:02}',12.25,6.98,.4,.23,9,GRAY)
    for s in slides:s.notes_slide.notes_text_frame.text=page['notes']
    preview=OUT/f'compact-slide-{number:02}.png';im.save(preview);previews.append(preview)
decks[0].save(OUT/'RL100_可编辑版.pptx')
decks[1].save(OUT/'RL100_图片版.pptx')
decks[1].save(OUT/'RL100_表格图片版.pptx')
sheet=Image.new('RGB',(1600,((len(previews)+1)//2)*470),'#E7EBEE')
for index,path in enumerate(previews):
    image=Image.open(path);image.thumbnail((770,434))
    sheet.paste(image,(15+(index%2)*800,15+(index//2)*470))
sheet.save(OUT/'预览总览.png')
(OUT/'compact-layout.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2)+'\n')
print('Built',len(pages),'slides per deck; 2 tables on pages 2, 3, 4; titles and captions remain editable.')
