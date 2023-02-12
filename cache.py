from urllib import request
from base64 import b64encode, b64decode
from time import sleep
import sys

with open('entries') as f:
    for i, line in enumerate(f.readlines()):
        url = line.strip()
        with request.urlopen(url) as res:
            filename_b64 = b64encode(url.encode('utf-8')).decode('utf-8')
            html = res.read().decode('utf-8')
            with open(f'entries/{filename_b64}', 'w') as f:
                f.write(html)
        sys.stdout.write(f'\r[{i}] test:{url}'.ljust(100))
