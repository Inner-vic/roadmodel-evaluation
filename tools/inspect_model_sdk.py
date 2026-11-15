import inspect
import contextlib
import io
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    import asset
print('asset_source',inspect.getfile(asset.Asset))
source=inspect.getsource(asset.Asset)
for marker in ['def __init__','def _get_s3_path']:
    start=source.find(marker)
    print(source[start:start+6500])
