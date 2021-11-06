#!/usr/bin/python3
from minio import Minio
from minio.error import S3Error
import json

"""
Last edited by   : Roy
Last edited time : 05/11/2021
"""

def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.

    login=json.load(open('credentials.json'))

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

    print(trainning_list[0])


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)