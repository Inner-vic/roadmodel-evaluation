"""Loopback-only board tunnel holding the workstation's existing bench lock."""
from __future__ import annotations

import json
import queue
import re
import shlex
import shutil
import subprocess
import threading

LOCK_WORKER = r'''
import fcntl,json,pathlib,sys
path=pathlib.Path(sys.argv[1])
if not path.is_absolute() or not path.parent.is_dir():raise ValueError('Configured bench lock directory is missing')
with path.open('a') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  print(json.dumps({'state':'bench_busy'}),flush=True);raise SystemExit(73)
 print(json.dumps({'state':'locked'}),flush=True)
 sys.stdin.buffer.read()
'''


class GatewayTunnel:
    def __init__(self, config):
        self.config = config
        self.process = None
        for key in ('ssh_alias', 'board_host'):
            if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', str(config.get(key, ''))):
                raise ValueError('Gateway requires a configured SSH alias and a plain board host')
        for key, default in [('local_port', 17222), ('board_port', 22)]:
            port = config.get(key, default)
            minimum=1024 if key=='local_port' else 1
            if type(port) is not int or not minimum <= port <= 65535:
                raise ValueError('Invalid gateway port')
        if not str(config.get('lock_file', '')).startswith('/'):
            raise ValueError('Gateway workstation lock_file is required')

    def __enter__(self):
        c = self.config
        command = [shutil.which('ssh.exe') or shutil.which('ssh') or 'ssh', '-o', 'BatchMode=yes',
                   '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
                   '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
                   '-L', f"127.0.0.1:{c.get('local_port',17222)}:{c['board_host']}:{c.get('board_port',22)}",
                   c['ssh_alias'], shlex.join([c.get('python','python3'), '-u', '-c', LOCK_WORKER, c['lock_file']])]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        messages = queue.Queue()
        threading.Thread(target=lambda: messages.put(self.process.stdout.readline()), daemon=True).start()
        try:
            line = messages.get(timeout=20)
            reply = json.loads(line) if line else {}
            if reply.get('state') != 'locked':
                raise ValueError('Board gateway unavailable or bench is busy; no evaluation started')
        except (queue.Empty, ValueError):
            self.__exit__(None, None, None)
            raise ValueError('Gateway lock/tunnel did not become ready; check existing SSH authentication and bench availability') from None
        return f"ssh root@127.0.0.1:{c.get('local_port',17222)}"

    def __exit__(self, *_):
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            if self.process.stdout:self.process.stdout.close()
            if self.process.stderr:self.process.stderr.close()
