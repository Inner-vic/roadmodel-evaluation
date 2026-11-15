"""Read retained TFRecord/protobuf scalar events without TensorFlow or model code."""
import collections,json,struct
from pathlib import Path
root=Path('evidence/model-evaluation/training-semantic-audit-20261027')
path=next((root/'original-output/tensorboard').glob('events.out.tfevents.*'))
def varint(b,i):
    n=s=0
    while True:
        x=b[i];i+=1;n|=(x&127)<<s
        if x<128:return n,i
        s+=7;assert s<=70
def fields(b):
    i=0
    while i<len(b):
        key,i=varint(b,i);num,wire=key>>3,key&7
        if wire==0:v,i=varint(b,i)
        elif wire==1:v=b[i:i+8];i+=8
        elif wire==2:
            size,i=varint(b,i);v=b[i:i+size];i+=size
        elif wire==5:v=b[i:i+4];i+=4
        else:raise ValueError('Unsupported protobuf wire')
        assert i<=len(b);yield num,wire,v
table=[]
for i in range(256):
    n=i
    for _ in range(8):n=(n>>1)^ (0x82f63b78 if n&1 else 0)
    table.append(n)
def crc(b):
    n=0xffffffff
    for x in b:n=table[(n^x)&255]^(n>>8)
    n^=0xffffffff
    return (((n>>15)|(n<<17))+0xa282ead8)&0xffffffff
raw=path.read_bytes();offset=0;events=0;scalars=collections.defaultdict(list);types=collections.defaultdict(collections.Counter)
while offset<len(raw):
    header=raw[offset:offset+8];length=struct.unpack('<Q',header)[0]
    assert length<16*1024*1024 and offset+16+length<=len(raw)
    assert crc(header)==struct.unpack_from('<I',raw,offset+8)[0]
    body=raw[offset+12:offset+12+length]
    assert crc(body)==struct.unpack_from('<I',raw,offset+12+length)[0]
    offset+=16+length;events+=1;step=0
    event=list(fields(body))
    for n,w,v in event:
        if n==2:step=v
    for n,w,v in event:
        if n!=5:continue
        for sn,sw,sv in fields(v):
            if sn!=1:continue
            entries=list(fields(sv));tag=next((v.decode('utf-8') for n,w,v in entries if n==1),'')
            for n,w,v in entries:
                if n==2 and w==5:scalars[tag].append({'step':step,'value':struct.unpack('<f',v)[0]});types[tag]['scalar']+=1
                if n in [4,5,6,8,9]:types[tag][str(n)]+=1
receipt={'source':str(path),'record_count':events,'all_tfrecord_checksums_verified':True,'tags':{k:dict(v) for k,v in sorted(types.items())},'scalar_series':dict(scalars)}
(root/'tensorboard-scalars.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'events':events,'tags':len(types),'class_or_distribution_tags':[k for k in types if any(x in k.lower() for x in ['cement','asphalt','class','hist','label','loss'])]},indent=2))
