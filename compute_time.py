import datetime
import argparse

parser = argparse.ArgumentParser(description='CIFAR ACCURACY')

parser.add_argument('--start', default='09-21-15:10:45',
                    help='model name.')
parser.add_argument('--end', default='09-22-15:11:45',
                    help='')

args = parser.parse_args()

start = datetime.datetime.strptime(args.start,'%m-%d-%H:%M:%S')
end = datetime.datetime.strptime(args.end,'%m-%d-%H:%M:%S')
print(start)
print(end)
delta = end - start
ans = delta.days * 24 * 60 + (delta.seconds / 60)
print(ans)
