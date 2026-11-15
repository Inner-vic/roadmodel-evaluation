"""Count object metadata for the user's exact asset; never download raw data."""
import collections
import contextlib
import datetime
import io
import json
from pathlib import Path
import sys
import time

from connect_active_entry import call


def main():
    original_stdout=sys.stdout
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            import asset
            p=Path('C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data/integrations.local.json')
            conn=next(c for c in json.loads(p.read_text(encoding='utf-8-sig'))['connections'] if c['id']=='lpai-sdk-closedloop-20261022')
            resource=asset.resource('datasets/smartchassisdata-v3/versions/26-10-20-1',
                config=asset.config(env='prod',bearer_file=conn['settings']['asset_bearer_file']))
            storage=resource._get_s3_path();client=resource._s3_client.get_s3_client(endpoint=storage['endpoint'])
            prefix=storage['path'].rstrip('/')+'/'
            assert storage['bucket']=='smartchassis-roadmodel-data' and prefix=='datasets/DataReorganized-V161-minidataset-test/'
            counts=collections.Counter();top=collections.Counter();scenes=set();frames=set();pages=0;total_bytes=0;examples=[]
            print('READ_ONLY_CLOUD_METADATA_LISTING_STARTED',file=original_stdout,flush=True)
            for page in client.get_paginator('list_objects_v2').paginate(Bucket=storage['bucket'],Prefix=prefix):
                pages+=1
                if pages>2000:raise ValueError('Metadata listing exceeded bounded scope')
                for obj in page.get('Contents',[]):
                    rel=obj['Key'][len(prefix):]
                    if not rel or rel.endswith('/'):continue
                    parts=rel.split('/');name=parts[-1];top[parts[0]]+=1;total_bytes+=obj['Size']
                    if len(parts)>=4 and parts[1]=='dataset' and parts[2].isdigit():scenes.add(parts[0]);frames.add((parts[0],parts[2]))
                    if name.startswith('img_label') and name.endswith('.json'):kind='visual_classification_GT_json'
                    elif name.endswith('.jpg') and '_f_ca_120' in name:kind='front_120_images'
                    elif name.endswith('.jpg') and ('f_ca_30' in name or 'f30' in name):kind='front_30_images'
                    elif name.startswith('occ_frame') and name.endswith('.npy'):kind='OCC_GT'
                    elif name.startswith('height_field') and name.endswith('.npy'):kind='height_field_GT'
                    elif name.startswith('lidar_original') and name.endswith('.npy'):kind='lidar_inputs'
                    elif name.startswith('T_frame') and name.endswith('.npy'):kind='pose_inputs'
                    elif name=='sensor_param.json':kind='sensor_parameters'
                    elif name.startswith('data_link') and name.endswith('.json'):kind='data_link_index'
                    else:kind='other'
                    counts[kind]+=1
                    if len(examples)<8:examples.append(rel)
                if pages%50==0:print(json.dumps({'pages':pages,'files_counted':sum(counts.values())}),file=original_stdout,flush=True)
            result={'checked_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'dataset':'smartchassisdata-v3','version':'26-10-20-1','source_uri':'s3://s3.bj.bcebos.com/smartchassis-roadmodel-data/'+prefix,
              'metadata_pages':pages,'file_counts':dict(counts),'total_files':sum(counts.values()),'total_bytes':total_bytes,
              'scene_directories':len(scenes),'frame_directories':len(frames),'top_level_directory_count':sum('/' not in k and k in scenes for k in top),
              'vision_named_top_level_directories':[k for k in top if 'vision' in k.lower()],
              'root_files':[k for k in top if k not in scenes][:20],'examples':examples,
              'raw_data_downloaded_bytes':0,'cloud_writes':0}
        code="import pathlib,json;p=pathlib.Path('/mnt/data/bench-eval/roadmodel/inspection/dataset-structure-20261103/cloud-inventory.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(%r);print(json.dumps(json.loads(p.read_text()),ensure_ascii=False))" % json.dumps(result,ensure_ascii=False,indent=2)
        print(call(code,timeout=45),flush=True)
    except Exception as error:
        response=getattr(error,'response',{})
        print(json.dumps({'state':'SOURCE_METADATA_ACCESS_FAILED','error_type':type(error).__name__,
            'http_status':response.get('ResponseMetadata',{}).get('HTTPStatusCode',getattr(error,'status',None)),
            'service_error_code':response.get('Error',{}).get('Code')}),flush=True)
        raise SystemExit(2)


if __name__=='__main__':main()
