import io, os
import argparse
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description="File Performance Testing Util")

parser.add_argument("command",help="Test to perform",choices=['read','write','readany'])
parser.add_argument("dir",help="Directory to use")

args = parser.parse_args()

def time(msg,size,f):
    print(msg,end='...')
    st = timer()
    f()
    el = timer()-st
    print("{} sec, {} Mb/sec".format(el,size/el/1024/1024))

def write_test(n,size):
    fn = "test_{}".format(size)+"_{}"
    buf = os.urandom(size)
    for i in range(n):
        with open(os.path.join(args.dir,fn.format(i)),'wb') as f:
            f.write(buf)

def read_test(n,size):
    fn = "test_{}".format(size)+"_{}"
    for i in range(n):
        with open(os.path.join(args.dir,fn.format(i)),'rb') as f:
            buf = bytearray(f.read())

def read_test(n=1000):
    sz = 0
    i = 0
    st = timer()
    for x in os.listdir(args.dir):
        with open(os.path.join(args.dir,x),'rb') as f:
            buf = bytearray(f.read())
            sz += len(buf)
            i += 1
        n-=1
        if n==0:
            break
    en = timer()-st
    print("{} secs, {} Mb/Sec, av file size: {} Mb".format(en,sz/1024/1024/en,sz/i/1024/1024))

if args.command == "read":
    time("1000 1k files",1024*1000,lambda: read_test(1000,1024))
    time("100 1M files",1024*1024*100,lambda: read_test(100,1024*1024))
    time("10 10M files",10*1024*1024*10,lambda: read_test(10,1024*1024*10))
    time("1 100M files",1*1024*1024*100,lambda: read_test(1,1024*1024*100))
elif args.command == "write":
    time("1000 1k files",1024*1000,lambda: write_test(1000,1024))
    time("100 1M files",1024*1024*100,lambda: write_test(100,1024*1024))
    time("10 10M files",10*1024*1024*10,lambda: write_test(10,1024*1024*10))
    time("1 100M files",1*1024*1024*100,lambda: write_test(1,1024*1024*100))
elif args.command == "readany":
    read_test()
    