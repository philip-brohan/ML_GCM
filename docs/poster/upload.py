!#/usr/bin/env python

# Upload poster files to S3

import boto3
s3 = boto3.resource('s3')

bucket_n='philip.brohan.org.big-files'
bucket_o = s3.Bucket(name=bucket_n)

def get_s3_time(fname):
    result=None
    for f in bucket_objects.filter(Prefix='ML_GCM')):
        
