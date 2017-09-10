# FDS upload
# https://github.com/XiaoMi/galaxy-fds-sdk-python
# install the fds-sdk package with
# pip install galaxy-fds-sdk
import os, sys
from fds import GalaxyFDSClient, GalaxyFDSClientException

# you should have set os environment variables for FDS
XIAOMI_ACCESS_KEY_ID=os.environ.get('XIAOMI_ACCESS_KEY_ID')
XIAOMI_SECRET_ACCESS_KEY=os.environ.get('XIAOMI_SECRET_ACCESS_KEY')

# create a random bucket name
bucket = os.environ.get('XIAOMI_FDS_DEFAULT_BUCKET')
if bucket is None:
    # create a random tag
    import hashlib
    tag = hashlib.md5(os.urandom(32)).hexdigest()[:8]
    bucket = "torcs-" + tag
    # WARNING: the following two won't do since child process cann't change
    # the settings for its parents
    #os.environ['XIAOMI_FDS_DEFAULT_BUCKET'] = bucket
    #os.system("export XIAOMI_FDS_DEFAULT_BUCKET='%s'"%bucket)
    # permanently for next restart
    with open(os.path.expanduser("~/.bashrc"), "a") as f: 
        f.write("export XIAOMI_FDS_DEFAULT_BUCKET='%s'\n"%bucket)
        print 'Warning: You must restart a terminal window to take into effect.'

# endpoint is read from ~/.config/xiaomi/config
#client = GalaxyFDSClient(XIAOMI_ACCESS_KEY_ID, XIAOMI_SECRET_ACCESS_KEY)
client = GalaxyFDSClient()

if not client.does_bucket_exist(bucket):
    try:
        print 'Create bucket ', bucket
        client.create_bucket(bucket)
    except GalaxyFDSClientException as e:
        print e.message

# no need to delete since duplicate objects will be replaced when put
# try:
#     client.delete_object(bucket, "tf_train-1.0.tar.gz")
# except GalaxyFDSClientException as e:
#     print e.message

tensorflow_model = 'TFTrainerPredictor-1.4.tar.gz'
keras_model = 'KerasTrainer-1.0.tar.gz'

try:
    with open('../dist/' + tensorflow_model, 'rb') as f:
        data = f.read()
        res = client.put_object(bucket, tensorflow_model, data)
        print 'Put Object: ', res.signature, res.expires
        client.set_public(bucket, tensorflow_model)
        print 'Set public'

    with open('../dist/' + keras_model, 'rb') as f:
        data = f.read()
        res = client.put_object(bucket, keras_model, data)
        print 'Put Object: ', res.signature, res.expires
        client.set_public(bucket, keras_model)
        print 'Set public'

except GalaxyFDSClientException as e:
    print e.message
