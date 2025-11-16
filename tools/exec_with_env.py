import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run a script with CUDA disabled in the child process')
    parser.add_argument('script', help='Path to the python script to run')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    parser.add_argument('--stdin', type=str, default=None, help='String to send to stdin')
    args = parser.parse_args()

    # Disable CUDA for the child process
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''

    cmd = [sys.executable, args.script] + args.args
    if args.stdin is not None:
        proc = subprocess.run(cmd, input=args.stdin.encode('utf-8'), env=env)
    else:
        proc = subprocess.run(cmd, env=env)
    return proc.returncode

if __name__ == '__main__':
    sys.exit(main())
