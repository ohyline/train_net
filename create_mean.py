import sys
caffe_root = '/home/mcl/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import caffe.proto.caffe_pb2 as caffe_proto
import numpy as np

def create_mean_file(filename, mean):
    """
    Create a binaryproto file with a single blob containing the given pixel-wise mean.
    Overwrites any existing file with the same name.
    :param filename: Filename to be used
    :type filename: str
    :param mean: Numpy array holding the mean
    :type mean: numpy.multiarray.ndarray
    """
    blob = caffe_proto.BlobProto()
    blob.shape.dim.extend(map(int, mean.shape))
    blob.data.extend(mean.flatten().tolist())
    # legacy
    blob.channels = mean.shape[0]
    blob.height = mean.shape[1]
    blob.width = mean.shape[2]
    blob.num = 1
    with open(filename, "wb") as f:
        f.write(blob.SerializeToString())

filename = 'mean_128.binaryproto'


mean = np.ones((3, 227, 227), dtype=np.float) * 128
create_mean_file(filename, mean)

mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(filename, 'rb').read())
mean_npy = caffe.io.blobproto_to_array(mean_blob)

print "done"

