#!/usr/bin/python3
from io import StringIO
from minio import Minio
from minio import error
from minio.error import S3Error
import json

"""
Last edited by   : Roy
Last edited time : 06/11/2021
Version Status: stable
TO DO: allow users to change the bucket and object # line 28
"""

def storage_connection(credential_path):
    """
    @ input : your credential_path
    @ output: python dictionary with all ASTs
    """

    login=json.load(open(str(credential_path)))

    client = Minio(
        "s3.csc.uvic.ca:9000",
        access_key=login['console'][0]['access_key'].replace(u'\xa0', u''),
        secret_key=login['console'][0]['secret_key'].replace(u'\xa0', u''),
    )

    test_obj=client.get_object("RawDataStorage","example.json")
    trainning_list={}
    ast_index=0

    for ast in test_obj:
        trainning_list[ast_index]=json.loads(ast.decode('utf-8'))
        ast_index+=1
    
    return trainning_list
   

     