#!/usr/bin/env python3
"""
完整功能：
- 扫描目录下 JPG/JPEG 图片
- 读取 EXIF GPS (WGS84)
- 计算每张照片的 3°带中央子午线
- WGS84 -> CGCS2000 投影
- WGS84 -> Beijing1954 / Xian1980 (地理 + 投影)
- WGS84 -> GCJ-02（用于高德地图）
- 生成高德地图 HTML，点击 Marker 弹出照片和坐标信息
"""

import os, sys, math, json, piexif
from PIL import Image
from pyproj import CRS, Transformer
from jinja2 import Template

# 高德地图 API key
AMAP_KEY = "83ef8037f56058de95c2037b9e97e8ef"

# ----------------- EXIF GPS 读取 -----------------
def dms_to_deg(dms):
    deg = dms[0][0]/dms[0][1]
    minu = dms[1][0]/dms[1][1]
    sec = dms[2][0]/dms[2][1]
    return deg + minu/60 + sec/3600

def extract_gps_from_exif(path):
    try:
        exif_dict = piexif.load(path)
        gps = exif_dict.get("GPS")
        if not gps: return None
        lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
        lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)
        lat = gps.get(piexif.GPSIFD.GPSLatitude)
        lon = gps.get(piexif.GPSIFD.GPSLongitude)
        if not (lat and lon and lat_ref and lon_ref): return None
        lat_deg = dms_to_deg(lat)
        lon_deg = dms_to_deg(lon)
        if lat_ref.decode().upper() == 'S': lat_deg = -lat_deg
        if lon_ref.decode().upper() == 'W': lon_deg = -lon_deg
        return (lat_deg, lon_deg)
    except:
        return None

# ----------------- GCJ-02 坐标转换 -----------------
def out_of_china(lon, lat):
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)

def transform_lat(x, y):
    ret = -100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*x*y + 0.2*math.sqrt(abs(x))
    ret += (20*math.sin(6*x*math.pi)+20*math.sin(2*x*math.pi))*2/3
    ret += (20*math.sin(y*math.pi)+40*math.sin(y/3*math.pi))*2/3
    ret += (160*math.sin(y/12*math.pi)+320*math.sin(y*math.pi/30))*2/3
    return ret

def transform_lon(x, y):
    ret = 300.0 + x + 2.0*y + 0.1*x*x + 0.1*x*y + 0.1*math.sqrt(abs(x))
    ret += (20*math.sin(6*x*math.pi)+20*math.sin(2*x*math.pi))*2/3
    ret += (20*math.sin(x*math.pi)+40*math.sin(x/3*math.pi))*2/3
    ret += (150*math.sin(x/12*math.pi)+300*math.sin(x/30*math.pi))*2/3
    return ret

def wgs84_to_gcj02(lon, lat):
    if out_of_china(lon, lat): return lon, lat
    a = 6378245.0; ee=0.006693421622965943
    dlat = transform_lat(lon-105, lat-35)
    dlng = transform_lon(lon-105, lat-35)
    radlat = lat/180*math.pi
    magic = math.sin(radlat); magic = 1-ee*magic*magic; sqrtmagic = math.sqrt(magic)
    dlat = (dlat*180)/((a*(1-ee))/ (magic*sqrtmagic)*math.pi)
    dlng = (dlng*180)/(a/sqrtmagic*math.cos(radlat)*math.pi)
    mglat = lat + dlat; mglng = lon + dlng
    return mglng, mglat

# ----------------- 中央子午线计算 -----------------
def central_meridian_for_longitude(lon_deg):
    zone = int(lon_deg/3)+1
    lon0 = zone*3
    return zone, lon0

# ----------------- 主流程 -----------------
def process_folder(folder_in, folder_out):
    os.makedirs(folder_out, exist_ok=True)
    photos=[]
    for fname in os.listdir(folder_in):
        if not fname.lower().endswith(('.jpg','.jpeg')): continue
        path=os.path.join(folder_in,fname)
        gps=extract_gps_from_exif(path)
        if not gps: 
            print(f"[WARN] {fname} 无 GPS 信息，跳过。")
            continue
        lat, lon = gps
        zone, lon0 = central_meridian_for_longitude(lon)

        # WGS84 -> CGCS2000 (tmerc, lon_0=central_meridian, ellps=GRS80)
        tmerc_proj = CRS.from_proj4(f"+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 +x_0=500000 +ellps=GRS80 +units=m +no_defs")
        transformer_to_cgcs2000 = Transformer.from_crs("EPSG:4326", tmerc_proj, always_xy=True)
        x_cgcs2000, y_cgcs2000 = transformer_to_cgcs2000.transform(lon, lat)

        # WGS84 -> Beijing1954 / Xian1980 地理
        try: tr_wgs_to_beijing=Transformer.from_crs("EPSG:4326","EPSG:4214",always_xy=True)
        except: tr_wgs_to_beijing=None
        try: tr_wgs_to_xian=Transformer.from_crs("EPSG:4326","EPSG:4610",always_xy=True)
        except: tr_wgs_to_xian=None
        lon_bj, lat_bj = tr_wgs_to_beijing.transform(lon,lat) if tr_wgs_to_beijing else (None,None)
        lon_xi, lat_xi = tr_wgs_to_xian.transform(lon,lat) if tr_wgs_to_xian else (None,None)

        # 对应中央子午线下 tmerc 投影
        bj_tmerc = CRS.from_proj4(f"+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 +x_0=500000 +ellps=krass +units=m +no_defs")
        xi_tmerc = CRS.from_proj4(f"+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 +x_0=500000 +ellps=IAG75 +units=m +no_defs")
        tr_wgs_to_bjproj=Transformer.from_crs("EPSG:4326",bj_tmerc,always_xy=True)
        tr_wgs_to_xiproj=Transformer.from_crs("EPSG:4326",xi_tmerc,always_xy=True)
        x_bjproj,y_bjproj=tr_wgs_to_bjproj.transform(lon,lat)
        x_xiproj,y_xiproj=tr_wgs_to_xiproj.transform(lon,lat)

        # WGS84 -> GCJ-02
        gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)

        # 生成缩略图 base64
        thumb_path = os.path.join(folder_out,"thumbs")
        os.makedirs(thumb_path,exist_ok=True)
        thumb_file=os.path.join(thumb_path,fname)
        try:
            im=Image.open(path); im.thumbnail((400,300)); im.save(thumb_file,"JPEG",quality=70)
            rel_thumb=os.path.relpath(thumb_file,folder_out)
        except: rel_thumb=None

        photos.append({
            "filename":fname,
            "orig_path":os.path.relpath(path,folder_out),
            "lat_wgs84":lat,"lon_wgs84":lon,
            "zone":zone,"central_meridian":lon0,
            "cgcs2000_xy":[x_cgcs2000,y_cgcs2000],
            "beijing_geo":[lat_bj,lon_bj],"xian_geo":[lat_xi,lon_xi],
            "beijing_proj_xy":[x_bjproj,y_bjproj],"xian_proj_xy":[x_xiproj,y_xiproj],
            "gcj02":[gcj_lat,gcj_lon],
            "thumb_rel":rel_thumb
        })
        print(f"[INFO] {fname} WGS84=({lat:.6f},{lon:.6f}) 中央子午线={lon0}° CGCS2000(x,y)=({x_cgcs2000:.2f},{y_cgcs2000:.2f})")

    # 生成 HTML
    html = render_html_map(photos)
    out_html = os.path.join(folder_out,"map_output.html")
    with open(out_html,"w",encoding="utf-8") as f: f.write(html)
    print(f"[DONE] 地图生成：{out_html}")

# ----------------- HTML 模板 -----------------
def render_html_map(photos):
    tmpl=Template("""
<!doctype html>
<html><head><meta charset="utf-8"><title>照片地图</title>
<meta name="viewport" content="initial-scale=1.0, width=device-width">
<style>html,body,#map{height:100%;margin:0;padding:0;}.popup-img{max-width:300px;max-height:200px;}.info{font-size:12px;}</style>
<script src="https://webapi.amap.com/maps?v=2.0&key={{ amap_key }}"></script>
</head><body><div id="map"></div>
<script>
const map=new AMap.Map('map',{zoom:5,center:[116.397428,39.90923]});
const photos={{ photos_json|safe }};
photos.forEach(p=>{
  const marker=new AMap.Marker({position:[p.gcj02[1],p.gcj02[0]],map:map,title:p.filename});
  let imgHtml=p.thumb_rel?('<img src="'+p.thumb_rel+'" class="popup-img"><br>'):'';
  let infoHtml='<div class="info"><b>'+p.filename+'</b><br/>WGS84:'+p.lat_wgs84.toFixed(6)+','+p.lon_wgs84.toFixed(6)+'<br/>中央子午线:'+p.central_meridian+'°(zone '+p.zone+')<br/>CGCS2000(x,y):'+p.cgcs2000_xy.map(v=>v.toFixed(2)).join(', ')+'<br/>Beijing-1954(geo):'+(p.beijing_geo[0]?p.beijing_geo[0].toFixed(6)+','+p.beijing_geo[1].toFixed(6):'N/A')+'<br/>Xian-1980(geo):'+(p.xian_geo[0]?p.xian_geo[0].toFixed(6)+','+p.xian_geo[1].toFixed(6):'N/A')+'</div>';
  const infoWindow=new AMap.InfoWindow({content:imgHtml+infoHtml,offset:new AMap.Pixel(0,-30)});
  marker.on('click',()=>infoWindow.open(map,marker.getPosition()));
});
if(photos.length>0){map.setCenter([photos[0].gcj02[1],photos[0].gcj02[0]]);map.setZoom(12);}
</script></body></html>
""")
    # JSON 序列化
    for p in photos:
        p['gcj02']=[p['gcj02'][0],p['gcj02'][1]]
        p['cgcs2000_xy']=[p['cgcs2000_xy'][0],p['cgcs2000_xy'][1]]
        p['beijing_geo']=[p['beijing_geo'][0] if p['beijing_geo'][0] else None,
                          p['beijing_geo'][1] if p['beijing_geo'][1] else None]
        p['xian_geo']=[p['xian_geo'][0] if p['xian_geo'][0] else None,
                        p['xian_geo'][1] if p['xian_geo'][1] else None]
        p['beijing_proj_xy']=[p['beijing_proj_xy'][0],p['beijing_proj_xy'][1]]
        p['xian_proj_xy']=[p['xian_proj_xy'][0],p['xian_proj_xy'][1]]
    return tmpl.render(amap_key=AMAP_KEY,photos_json=json.dumps(photos,ensure_ascii=False))

# ----------------- CLI -----------------
if __name__=="__main__":
    if len(sys.argv)<3:
        print("用法: python script.py /path/to/photos /path/to/output_folder")
        sys.exit(1)
    folder_in=sys.argv[1]
    folder_out=sys.argv[2]
    process_folder(folder_in,folder_out)
