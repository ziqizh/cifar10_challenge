import datetime
import argparse

parser = argparse.ArgumentParser(description='CIFAR ACCURACY')

parser.add_argument('--start', default='09-21-05:24:37',
                    help='model name.')
parser.add_argument('--end', default='09-21-15:11:45',
                    help='')

args = parser.parse_args()

start = datetime.datetime.strptime(args.start,'%m-%d-%H:%M:%S')
end = datetime.datetime.strptime(args.end,'%m-%d-%H:%M:%S')
delta = end - start
print(delta.seconds / 60)
