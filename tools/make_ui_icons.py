"""Generate compact pixel-sized line icons; development-only Pillow dependency."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

folder = Path(__file__).resolve().parents[1] / 'assets' / 'icons'
folder.mkdir(exist_ok=True)
scale = 4
for name in ('settings', 'copy', 'more', 'refresh', 'close', 'minimize', 'check', 'send', 'microphone', 'microphone-active', 'waveform', 'add'):
    size = 24 if name in ('send', 'microphone', 'microphone-active', 'refresh', 'waveform', 'add') else 20
    im = Image.new('RGBA', (size*scale,size*scale))
    d = ImageDraw.Draw(im)
    color = '#007ACC'
    def line(points):
        d.line([(round(x*scale),round(y*scale)) for x,y in points],fill=color,width=round(1.5*scale),joint='curve')
    def rect(box):
        d.rounded_rectangle(tuple(round(v*scale) for v in box),radius=scale,outline=color,width=round(1.5*scale))
    if name == 'send':
        # Tk does not display SVG directly; rasterize the supplied straight-line
        # paths at 4x resolution, preserving their round caps/joins and source SVG.
        for path in ET.parse(folder / 'send.svg').getroot():
            values = list(map(float, re.findall(r'-?\d+(?:\.\d+)?', path.attrib['d'])))
            points = [(x*size*scale/48, y*size*scale/48) for x,y in zip(values[::2], values[1::2])]
            stroke = float(path.attrib['stroke-width'])*size*scale/48
            ink = path.attrib['stroke']
            d.line(points, fill=ink, width=round(stroke), joint='curve')
            for x,y in points:
                r = stroke/2
                d.ellipse((x-r,y-r,x+r,y+r), fill=ink)
    elif name.startswith('microphone'):
        # 48-unit SVG geometry, rendered at 4x the final size for smooth edges.
        unit = size*scale/48
        ink = '#D94A45' if name.endswith('active') else '#007ACC'
        stroke = round(4*unit)
        d.rounded_rectangle(tuple(v*unit for v in (17,4,31,31)), radius=7*unit,
                            outline=ink, width=stroke)
        d.arc(tuple(v*unit for v in (9,8,39,38)), start=0, end=180, fill=ink, width=stroke)
        d.line([(24*unit,38*unit),(24*unit,44*unit)], fill=ink, width=stroke)
        for x,y in ((9,23),(39,23),(24,38),(24,44)):
            d.ellipse(((x-2)*unit,(y-2)*unit,(x+2)*unit,(y+2)*unit), fill=ink)
    elif name == 'settings':
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
    elif name in ('refresh', 'waveform', 'add'):
        # Sample the original SVG's cubic Beziers (M/L/V/C paths) directly.
        unit = size*scale/48
        for path in ET.parse(folder / f'{name}.svg').getroot():
            tokens = iter(re.findall(r'[MLVCZ]|-?\d+(?:\.\d+)?', path.attrib['d']))
            strokes, points = [], []
            for command in tokens:
                if command == 'M':
                    if points:
                        strokes.append(points)
                    points = [(float(next(tokens)), float(next(tokens)))]
                elif command == 'L':
                    points.append((float(next(tokens)), float(next(tokens))))
                elif command == 'V':
                    points.append((points[-1][0], float(next(tokens))))
                elif command == 'Z':
                    points.append(points[0])
                elif command == 'C':
                    p0 = points[-1]
                    p1, p2, p3 = [(float(next(tokens)), float(next(tokens))) for _ in range(3)]
                    for i in range(1, 41):
                        t = i/40
                        points.append(tuple((1-t)**3*p0[a] + 3*(1-t)**2*t*p1[a] +
                                            3*(1-t)*t*t*p2[a] + t**3*p3[a] for a in (0,1)))
                else:
                    raise ValueError('Unsupported refresh SVG command: ' + command)
            strokes.append(points)
            ink = path.attrib['stroke']
            stroke = float(path.attrib['stroke-width'])*unit
            for points in strokes:
                d.line([(x*unit,y*unit) for x,y in points], fill=ink, width=round(stroke), joint='curve')
                for x,y in (points[0],points[-1]):
                    r = stroke/2
                    d.ellipse((x*unit-r,y*unit-r,x*unit+r,y*unit+r), fill=ink)
    elif name == 'close':
        line([(5,5),(15,15)])
        line([(15,5),(5,15)])
    elif name == 'minimize':
        line([(4,10),(16,10)])
    elif name == 'check':
        line([(4,10),(8,14),(16,5)])
    im.resize((size,size),Image.Resampling.LANCZOS).save(folder / f'{name}.png')
