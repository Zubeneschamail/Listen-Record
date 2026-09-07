"""Generate compact pixel-sized line icons; development-only Pillow dependency."""
from pathlib import Path
from PIL import Image, ImageDraw

folder = Path(__file__).resolve().parents[1] / 'assets' / 'icons'
folder.mkdir(exist_ok=True)
scale = 4
for name in ('settings', 'copy', 'more', 'refresh', 'close', 'minimize', 'check'):
    im = Image.new('RGBA', (20*scale,20*scale))
    d = ImageDraw.Draw(im)
    color = '#007ACC'
    def line(points):
        d.line([(round(x*scale),round(y*scale)) for x,y in points],fill=color,width=round(1.5*scale),joint='curve')
    def rect(box):
        d.rounded_rectangle(tuple(round(v*scale) for v in box),radius=scale,outline=color,width=round(1.5*scale))
    if name == 'settings':
        for y, x in ((5,7),(10,13),(15,8)):
            line([(3,y),(x-2,y)])
            line([(x+2,y),(17,y)])
            d.ellipse(((x-2)*scale,(y-2)*scale,(x+2)*scale,(y+2)*scale),outline=color,width=round(1.5*scale))
    elif name == 'copy':
        line([(6,13),(3,13),(3,3),(13,3),(13,6)])
        rect((7,7,17,17))
    elif name == 'more':
        for x in (4,10,16):
            d.ellipse(((x-1)*scale,9*scale,(x+1)*scale,11*scale),fill=color)
    elif name == 'refresh':
        d.arc((3*scale,3*scale,17*scale,17*scale),start=35,end=315,fill=color,width=round(1.5*scale))
        line([(12,5),(17,5),(17,0.8)])
    elif name == 'close':
        line([(5,5),(15,15)])
        line([(15,5),(5,15)])
    elif name == 'minimize':
        line([(4,10),(16,10)])
    elif name == 'check':
        line([(4,10),(8,14),(16,5)])
    im.resize((20,20),Image.Resampling.LANCZOS).save(folder / f'{name}.png')
