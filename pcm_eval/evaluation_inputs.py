"""Public card inputs: one delivery and one dataset for every evaluation head."""
from .dataset_mount import dataset_reference
from .model_ticket import inputs as model_inputs

FIELDS = ('model_delivery_url', 'dataset_url')


def validate_inputs(form):
    if not isinstance(form, dict) or set(form) != set(FIELDS):
        raise ValueError('请填写模型交付链接和评测数据集链接两个字段')
    if any(not isinstance(v, str) or not v.strip() or len(v) > 1000
           or any(c in v for c in '\r\n\x00') for v in form.values()):
        raise ValueError('两个链接均不能为空，每项限1000字符且不能换行')
    value = {k: form[k].strip() for k in FIELDS}
    model_inputs({'model_delivery_url': value['model_delivery_url'], 'dataset_ref': value['dataset_url']})
    for key in FIELDS[1:]:
        dataset_reference(value[key])
    return value
