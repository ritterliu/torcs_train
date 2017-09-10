import glob
import os
import sys

from fds import GalaxyFDSClient, GalaxyFDSClientException
from fds.model.fds_object_metadata import  FDSObjectMetadata

# endpoint is auto-read from ~/.config/xiaomi/config
client = GalaxyFDSClient()

bucket = os.environ.get('XIAOMI_FDS_DEFAULT_BUCKET') or 'johndoe'
#LOG_DIR =  '/home/mi/Documents/github/changbinglin/aicontest/logs'
DIRS = ['camera_data/raw_train', 'camera_data/raw_val', 'keras_model', 'tensorflow_model']

if len(sys.argv) > 1:
    LOG_DIR = sys.argv[1]

metadata = FDSObjectMetadata()
metadata.add_header('x-xiaomi-meta-mode', '33188') # give rights: rw-r--r--
try:
    for directory in DIRS:
        for log in glob.glob(directory + '/*'):
            if os.path.isfile(log):
                print log.split('/')[-1]
                if not client.does_object_exists(bucket, log):
                    with open(log, 'r') as f:
                        data = f.read()
                        #path_to = '/'.join(LOG_DIR.split('/')[-3:])
                        res = client.put_object(bucket, log, data, metadata)
                        print 'Put Object: ', res.signature, res.expires
                        client.set_public(bucket, log)
                        print 'Set public', log

except GalaxyFDSClientException as e:
    print e.message
