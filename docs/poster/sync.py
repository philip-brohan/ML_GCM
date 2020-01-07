!#/usr/bin/env python

# Sync file to S3

import boto3
s3 = boto3.client('s3')
import datetime
import pytz
import os.path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bucket", help="S3 bucket",
                    type=str,required=False,
                    default='philip.brohan.org.big-files')
parser.add_argument("--prefix", help="Bucket sub-dir",
                    type=str,required=True)
parser.add_argument("--name", help="File name",
                    type=str,required=True)
args = parser.parse_args()

def get_from_s3():
    paginator = s3.get_paginator('list_objects')
    page_iterator = paginator.paginate(Bucket=args.bucket)
    for bucket_p in page_iterator: 
        for f in bucket_p['Contents']:
            if f['Key'] == "%s/%s" % (args.prefix,args.name):
                return f
    return None

def get_from_local():
    try:
        mt=os.path.getmtime(args.name)
    except FileNotFoundError:
        return None
    dt=datetime.utcfromtimestamp(mt)
    dt=dt.replace(pytz.UTC)
    sz=os.path.getsize(args.name)
    return {'Size':sz,'LastModified':dt}

def upload:
    s3.upload_file(
        args.name, args.bucket, '%s/%s' % (args.prefix,args.name),
        ExtraArgs={'ACL': 'public-read'},
        Callback=ProgressPercentage(args.name)
    )
    
def download:
    sr = boto3.resource('s3')
    s3.Bucket(args.bucket).download_file('%s/%s' % (args.prefix,args.name),
                                         args.name)


class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()



s3d=get_from_s3()
locald=get_from_local()

if s3d is None:
    if locald is None:
        raise ValueException("File not found at either end")
    else:
        upload()
else:
    if locald is None:
        download()
    else:
        if ( locald['Size']!=s3d['Size'] and
             locald['LastModified']>s3d['LastModified'] ):
            upload()
        if ( locald['Size']!=s3d['Size'] and
             locald['LastModified']<s3d['LastModified'] ):
            download()
