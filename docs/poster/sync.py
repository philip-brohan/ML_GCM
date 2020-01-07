#!/usr/bin/env python

# Sync file to S3

import boto3
s3 = boto3.client('s3')
import datetime
import pytz
import os.path
import threading
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bucket", help="S3 bucket",
                    type=str,required=False,
                    default='philip.brohan.org.big-files')
parser.add_argument("--prefix", help="Bucket sub-dir",
                    type=str,required=False,
                    default=None)
parser.add_argument("--name", help="File name",
                    type=str,required=True)
args = parser.parse_args()

key=args.name
if args.prefix is not None:
    key = "%s/%s" % (args.prefix,args.name)

def get_from_s3():
    paginator = s3.get_paginator('list_objects')
    page_iterator = paginator.paginate(Bucket=args.bucket)
    for bucket_p in page_iterator: 
        try:
            for f in bucket_p['Contents']:
                if f['Key'] == key:
                    return f
        except KeyError:
            return None
    return None

def get_from_local():
    try:
        mt=os.path.getmtime(args.name)
    except FileNotFoundError:
        return None
    dt=datetime.datetime.fromtimestamp(mt,pytz.UTC)
    #dt=dt.replace(pytz.UTC)
    sz=os.path.getsize(args.name)
    return {'Size':sz,'LastModified':dt}

def upload():
    print("Uploading")
    s3.upload_file(
        args.name, args.bucket, key,
        ExtraArgs={'ACL': 'public-read'},
        Callback=ProgressPercentage(args.name,locald['Size'])
    )
    
def download():
    print("Downloading")
    sr = boto3.resource('s3')
    sr.Bucket(args.bucket).download_file(key,args.name,
                                         Callback=ProgressPercentage(
                                             args.name,s3d['Size']))


class ProgressPercentage(object):

    def __init__(self, filename,filesize):
        self._filename = filename
        self._size = filesize
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
        raise Exception("File not found at either end")
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
