#!/usr/bin/python3
from minio import Minio
import json

"""
Last edited by   : Roy
Last edited time : 14/11/2021
Version Status: stable
TO DO: None
"""


def storage_connection(credential_path,object):
    """
    @ input : your credential_path
    @ output: python dictionary with all ASTs
    """

    login = json.load(open(str(credential_path)))

    client = Minio(
        "s3.csc.uvic.ca:9000",
        access_key=login['console'][0]['access_key'].replace(u'\xa0', u''),
        secret_key=login['console'][0]['secret_key'].replace(u'\xa0', u''),
    )

    test_obj = client.get_object("RawDataStorage", object)
    training_list = {}
    ast_index = 0

    for ast in test_obj:
        training_list[ast_index] = json.loads(ast.decode('ISO-8859-1'))
        ast_index += 1

    return training_list