# FDS download
# https://github.com/XiaoMi/galaxy-fds-sdk-python
# install the fds-sdk package with
# pip install galaxy-fds-sdk
import os, sys
from fds import GalaxyFDSClient, GalaxyFDSClientException

# you should have set os environment variables for FDS
XIAOMI_ACCESS_KEY_ID=os.environ.get('XIAOMI_ACCESS_KEY_ID')
XIAOMI_SECRET_ACCESS_KEY=os.environ.get('XIAOMI_SECRET_ACCESS_KEY')

# get bucket name
bucket = os.environ.get('XIAOMI_FDS_DEFAULT_BUCKET')
if bucket is None:
    print 'Error: Bucket not found.'
    sys.exit()

# endpoint is read from ~/.config/xiaomi/config
client = GalaxyFDSClient()
model_types = ['keras_model/', 'tensorflow_model/']
for mt in model_types:
    object_list = client.list_objects(bucket, mt)
    print 'found ', object_list
    for obj in object_list.objects:
        print 'download', obj.object_name
        try:
            client.download_object(bucket, obj.object_name, obj.object_name)
        except GalaxyFDSClientException as e:
            print e.message
