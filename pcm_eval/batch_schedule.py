"""Ordered power-of-two native batches with explicit retained-result boundaries."""


def groups(indices, width, retained=()):
    if type(width) is not int or width not in (1,2,4,8):
        raise ValueError('Batch width must be 1, 2, 4 or 8')
    indices=list(indices);retained=set(retained)
    if any(type(i) is not int or i<0 for i in indices) or indices!=sorted(set(indices)):
        raise ValueError('Exact ordered unique indices required')
    pending=[]
    def drain():
        while pending:
            size=max(n for n in (1,2,4,8) if n<=min(width,len(pending)))
            batch=pending[:size];del pending[:size];yield {'mode':'infer','indices':batch}
    for i in indices:
        if i in retained:
            yield from drain()
            yield {'mode':'adopt','indices':[i]}
        else:
            pending.append(i)
            if len(pending)==width:yield from drain()
    yield from drain()
