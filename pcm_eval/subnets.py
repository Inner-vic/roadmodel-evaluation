"""Explicit five-subnet contract; no inference and no compiler binary rewriting."""
from copy import deepcopy
from math import prod
from pathlib import PurePosixPath
import re

DTYPE_BYTES={'float16':2,'float32':4,'int32':4,'int64':8,'uint8':1}


def tensor_port(path):
    name=PurePosixPath(path).name
    match=re.fullmatch(r'(.+)\.([0-9]+(?:x[0-9]+)*)\.(float16|float32|int32|int64|uint8)\.bin',name)
    if not match:raise ValueError('Unrecognized tensor port: '+name)
    shape=[int(x) for x in match[2].split('x')]
    if any(x<1 for x in shape):raise ValueError('Invalid tensor dimension')
    return {'tensor':match[1],'shape':shape,'dtype':match[3],'filename':name,
            'nbytes':prod(shape)*DTYPE_BYTES[match[3]]}


def external_filename(port):
    # The original whole-network exporter includes the lidar_backbone module prefix.
    # Standalone sub1 was exported without that prefix; names/types/shapes are fixed.
    name=port['filename']
    if re.fullmatch(r'@pillar_layer@voxel_layer@LpexVoxelizationTRT_output_[012]\..+\.bin',name):
        return '@lidar_backbone'+name
    return name


def chain_contract(infos):
    if set(infos)!={0,1,2,3,4}:raise ValueError('Exactly sub0..sub4 are required')
    producers={};stages=[];external={};edges=[]
    for index in range(5):
        info=infos[index]
        if not info.get('elf_path'):raise ValueError('No ELF entry point')
        inputs=[]
        for entry in info['input']:
            paths=entry.get('data_paths')
            if not isinstance(paths,list) or len(paths)!=1:raise ValueError('Use the original single-frame port contract')
            port=tensor_port(paths[0]);source=producers.get(port['tensor'])
            if source:
                prior,output=source
                if any(output[k]!=port[k] for k in ('shape','dtype','nbytes')):
                    raise ValueError('Subnet boundary changes shape or dtype: '+port['tensor'])
                port['producer']=prior;edges.append({'from':prior,'to':index,'tensor':port['tensor']})
            else:
                port['external_filename']=external_filename(port)
                old=external.get(port['external_filename'])
                if old and any(old[k]!=port[k] for k in ('shape','dtype')):raise ValueError('Inconsistent external tensor')
                external[port['external_filename']]=port
            inputs.append(port)
        outputs=[]
        for entry in info['output']:
            paths=entry.get('golden_paths')
            if not isinstance(paths,list) or len(paths)!=1:raise ValueError('Output needs an original shape/type declaration')
            port=tensor_port(paths[0])
            if port['tensor'] in producers:raise ValueError('Ambiguous tensor producer')
            port['declared_data_path']=entry['data_path'];outputs.append(port)
        stages.append({'stage':index,'inputs':inputs,'outputs':outputs})
        for port in outputs:producers[port['tensor']]=(index,port)
    if len(external)!=11:raise ValueError('Unexpected whole-network external input count')
    return {'schema':'roadmodel.subnet-chain/v1','stages':stages,'external_inputs':list(external.values()),'edges':edges}


def stage_info(template,stage,root):
    root=PurePosixPath(root)
    if not root.is_absolute() or '..' in root.parts:raise ValueError('Absolute isolated board root required')
    result=deepcopy(template)
    for original,port in zip(result['input'],stage['inputs'],strict=True):
        path=(root/f"sub{port['producer']}"/'export'/port['filename'] if 'producer' in port
              else root/'inputs'/port['external_filename'])
        original['data_paths']=[str(path)]
    for original,port in zip(result['output'],stage['outputs'],strict=True):
        original['data_path']='./output/'+PurePosixPath(port['declared_data_path']).name
        original.pop('data_paths',None);original['golden_paths']=[]
        if 'fm_golden_paths' in original:original['fm_golden_paths']=[]
    result.update(infer_num=1,buffer_batch_num=1,do_infer=True,do_compare=False,do_dump=True)
    return result
