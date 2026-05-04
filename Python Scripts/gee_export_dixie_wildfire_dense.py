from __future__ import annotations
import argparse, datetime as dt, json
from pathlib import Path
GEE_PROJECT="rapid-being-484808-t1"
ROI=[-121.65,39.55,-120.85,40.40]
EXPORT_FOLDER="BITS_exports"; EXPORT_PREFIX="dixie_wildfire_dense_stack"; METADATA_JSON="dixie_wildfire_dense_metadata.json"
WINDOWS=[("pre_202104","2021-04-10","2021-04-30"),("pre_202105","2021-05-10","2021-05-31"),("pre_202106","2021-06-10","2021-06-30"),("fire_202107","2021-07-10","2021-07-31"),("fire_202108a","2021-08-01","2021-08-15"),("fire_202108b","2021-08-16","2021-08-31"),("post_202109","2021-09-10","2021-09-30"),("post_202110","2021-10-05","2021-10-31"),("rec_202204","2022-04-10","2022-04-30"),("rec_202206","2022-06-10","2022-06-30"),("rec_202304","2023-04-10","2023-04-30"),("rec_202306","2023-06-10","2023-06-30"),("rec_202404","2024-04-10","2024-04-30"),("rec_202406","2024-06-10","2024-06-30"),("rec_202504","2025-04-10","2025-04-30"),("rec_202506","2025-06-10","2025-06-30")]
BANDS=["sur_refl_b01","sur_refl_b02","sur_refl_b03","sur_refl_b04","sur_refl_b05","sur_refl_b06","sur_refl_b07","state_1km"]
REN=["b01","b02","b03","b04","b05","b06","b07","state_1km"]
def _date(ms:int)->str: return dt.datetime.utcfromtimestamp(ms/1000).strftime('%Y%m%d')
def export_to_drive():
    import ee; ee.Initialize(project=GEE_PROJECT); roi=ee.Geometry.Rectangle(ROI)
    base=ee.ImageCollection('MODIS/061/MOD09GA').filterBounds(roi).select(BANDS)
    def add_bad(img):
        s=img.select('state_1km'); cloud=s.bitwiseAnd(3); shadow=s.rightShift(2).bitwiseAnd(1); aer=s.rightShift(6).bitwiseAnd(3); cir=s.rightShift(8).bitwiseAnd(3)
        bad=cloud.eq(1).Or(shadow.eq(1)).Or(cir.eq(3)).Or(aer.eq(3))
        frac=bad.reduceRegion(reducer=ee.Reducer.mean(),geometry=roi,scale=1000,maxPixels=1e8,bestEffort=True).get('state_1km')
        return img.set('bad_frac',frac)
    imgs=[]; labels=[]; dates=[]; bads=[]
    for lab,st,en in WINDOWS:
        col=base.filterDate(st,en).map(add_bad).sort('bad_frac'); print(lab,'found',int(col.size().getInfo()))
        first=ee.Image(col.first()); imgs.append(first.clip(roi).rename([f'{lab}_{b}' for b in REN]).toInt32())
        dates.append(_date(int(first.get('system:time_start').getInfo()))); bf=first.get('bad_frac').getInfo(); bads.append(float(bf) if bf is not None else None); labels.append(lab)
        print(' selected',dates[-1],'bad_frac=',bads[-1])
    mod09=ee.Image.cat(imgs).toInt32()
    burn=ee.ImageCollection('MODIS/061/MCD64A1').filterDate('2021-07-01','2021-11-01').filterBounds(roi).select('BurnDate').max().clip(roi).rename('burn_2021').unmask(0).toInt32()
    stack=ee.Image.cat([mod09,burn]).toInt32()
    Path(METADATA_JSON).write_text(json.dumps({'scenario':'Dixie Fire dense wildfire recovery','roi':ROI,'mod09ga_collection':'MODIS/061/MOD09GA','mod09ga_doi':'10.5067/MODIS/MOD09GA.061','mcd64a1_collection':'MODIS/061/MCD64A1','window_labels':labels,'selected_dates':dates,'selected_bad_fracs':bads,'renamed_bands':REN,'windows':WINDOWS,'export_prefix':EXPORT_PREFIX},indent=2))
    task=ee.batch.Export.image.toDrive(image=stack,description='BITS_Dixie_dense_MOD09GA_MCD64A1_INT32',folder=EXPORT_FOLDER,fileNamePrefix=EXPORT_PREFIX,region=roi,scale=500,maxPixels=1e9); task.start(); print('Task ID:',task.id)
def convert(tif,npz,metadata=METADATA_JSON):
    import numpy as np, rasterio
    meta=json.loads(Path(metadata).read_text()); T=len(meta['window_labels']); nb=len(REN); exp=T*nb+1
    with rasterio.open(tif) as src: arr=src.read(); print('GeoTIFF shape:',arr.shape,'CRS:',src.crs)
    if arr.shape[0]!=exp: raise RuntimeError(f'Expected {exp} bands, got {arr.shape[0]}')
    mod09=arr[:T*nb].reshape(T,nb,arr.shape[1],arr.shape[2]); data={}
    for i,n in enumerate(REN): data[n]=mod09[:,i,:,:]
    data['burn_2021']=arr[-1]; data['dates']=np.array(meta['selected_dates'],dtype='<U16'); data['window_labels']=np.array(meta['window_labels'],dtype='<U32'); data['bad_frac']=np.array(meta['selected_bad_fracs'],dtype=np.float32); data['roi']=np.array(meta['roi'],dtype=np.float32)
    np.savez(npz,**data); print('Saved',npz)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--export',action='store_true'); p.add_argument('--convert',action='store_true'); p.add_argument('--tif',default=f'{EXPORT_PREFIX}.tif'); p.add_argument('--npz',default='dixie_wildfire_dense.npz'); p.add_argument('--metadata-json',default=METADATA_JSON); a=p.parse_args()
    if a.export: export_to_drive()
    if a.convert: convert(a.tif,a.npz,a.metadata_json)
    if not a.export and not a.convert: p.print_help()
if __name__=='__main__': main()