import h5py
import numpy as np
import argparse
import os
# os.environ['GLOG_minloglevel'] = '3'
import caffe
from PIL import Image
from typing import Union, List
from scipy.ndimage.filters import median_filter


def ensure_ui8_rgb(arr: np.array):
    buff = np.squeeze(arr)
    buff *= 255. / np.max(buff)
    buff = buff.astype('uint8')
    if len(buff.shape) == 2:
        buff = buff[:, :, np.newaxis]
    elif len(buff.shape) == 3:
        if buff.shape[0] in [1, 3]:
            buff = np.moveaxis(buff, 0, -1)
    if buff.shape[2] == 1:
        buff = np.repeat(buff, 3, axis=2)
    return buff


def insert_at(canvas, im, x, y):
    canvas[y:y+im.shape[0], x:x+im.shape[1]] = im
    return canvas


def h_stack(arrays: List[np.array], space: Union[int, float]=0.1, fill: int = 127):
   if 0 < space <1:
      space = int(np.median([e.shape[1] for e in arrays])*space)
   canvas_width = int(sum([e.shape[1] for e in arrays])) + (len(arrays)-1)*space
   canvas_height = int(max([e.shape[0] for e in arrays]))
   canvas = np.ones(shape=(canvas_height, canvas_width, 3), dtype='uint8')*fill
   start_x = 0
   for arr in arrays:
       start_y = (canvas_height - arr.shape[0]) // 2
       canvas = insert_at(canvas, arr, start_x, start_y)
       start_x += arr.shape[1] + space
   return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an original map or tiles in a hdf5 file.')
    parser.add_argument('input', type=str,
                        help='file to read data from')
    parser.add_argument('--network', type=str, default='deploy.prototxt',
                        help='the network definition to use')

    parser.add_argument('--model', type=str,
                        help='the caffemodel file to use', default='.')

    args = parser.parse_args()

    source_file = args.input
    model = args.model
    network = args.network
    if not (os.path.exists(source_file) and os.path.isfile(source_file)):
        raise ValueError("Could not read file '{}' as input".format(source_file))
    else:
        source_file = os.path.abspath(source_file)
        print("Reading input data from {}".format(source_file))

    if not (os.path.exists(network) and os.path.isfile(network)):
        raise ValueError("Could not read file '{}' as network".format(network))
    else:
        network = os.path.abspath(network)
        print("Reading network layout from {}".format(network))

    if not os.path.exists(model):
        raise ValueError("Could not read trained model from '{}'".format(model))
    if os.path.isdir(model):
        model = os.path.abspath(model)
        print("Searching '{}' for trained models".format(model))

        model_candidates = sorted([f for f in os.listdir(model) if f.endswith(
                '.caffemodel')])
        print("Model file '{}' seems to be the latest, using it.".format(
            model_candidates[-1]))
        model = os.path.join(model, model_candidates[-1])
    else:
        model = os.path.abspath(model)
        print("Reading trained model from '{}'".format(model))
caffe.set_mode_gpu()
n = caffe.Net(network, caffe.TEST, weights=model)
input_blob = n.inputs[0]
output_blob = n.outputs[0]
input_shape=n.blobs[input_blob].data.shape
output_shape = n.blobs[output_blob].data.shape
if not len(input_shape) == 4:
    raise ValueError("Can not determine input size from shape: {}".format(input_shape))
if input_shape[1] == 3:
    print("Expecting color input")
elif input_shape[1] == 1:
    print("Expecting grey level input")
else:
    raise ValueError("Can not work with input dimensionality '{}'".format(
        input_shape[1]))
receptive_height = input_shape[2]
receptive_width = input_shape[3]

if not len(output_shape) == 4:
    raise ValueError("Can not determine output size from shape: {}".format(
        output_shape))

print("Producing {} channels as output".format(output_shape[1]))
output_height = output_shape[2]
output_width = output_shape[3]

pad_height = (receptive_height - output_height) // 2
pad_width = (receptive_width - output_width) // 2

if source_file.endswith('.hdf5'):
    with h5py.File(source_file, "r") as f:
        data = f["data"][...]
        label = f["labels"][...]

    for i in range(100):
        input = np.flip(data[i, :, :, :][np.newaxis, :, :, :], axis=(2, 3))
        out = n.forward_all(data=input)["out"][0]
        # out[out <= 0.] = 0.
        # out[out > 0.] = 1.
    #    filtered = median_filter(out, size=3)
        canvas = h_stack([ensure_ui8_rgb(input),
                          255-ensure_ui8_rgb(out),
                          ensure_ui8_rgb(np.flip(label[i], axis=(-2, -1)))])
        im = Image.fromarray(canvas)
        im.save('out_dist-{}.png'.format(i))
    #    gt = Image.fromarray(ensure_ui8_rgb(label[i]))
    #    gt.save('gt-{}.png'.format(i))
    #    input = np.moveaxis(data[i],0 , -1)*255
    #    input=Image.fromarray(ensure_ui8_rgb(input))
    #    input.save('in-{}.png'.format(i))
else:
    im = np.array(Image.open(source_file), dtype=np.float32) / 255.
    if len(im.shape) == 3 and im.shape[2] == 4:
        im = im[:, :, :3]  # remove alpha
    im = np.rollaxis(im, -1)  # make [Chan, Height, Width] from [Height, Width, Chan]
    out = np.zeros(shape=(output_shape[1], im.shape[1], im.shape[2]))
    pos_x = 0
    pos_y = 0
    while pos_y < im.shape[1]:
        while pos_x < im.shape[2]:
            inp = im[np.newaxis, :,
                  pos_y:pos_y + receptive_height,
                  pos_x:pos_x + receptive_width]
            # n.blobs[input_blob].data[...] = inp
            # n.forward()
            out[:,
                pos_y + pad_height:pos_y + pad_height + output_height,
                pos_x + pad_width:pos_x + pad_width + output_width] = n.forward_all(data=inp)["out"][0]

            if pos_x == im.shape[2] - receptive_width:
                pos_x = im.shape[2]
            else:
                pos_x += receptive_width
                if pos_x > im.shape[2] - receptive_width:
                    pos_x = im.shape[2] - receptive_width
        pos_x = 0
        if pos_y == im.shape[1] - receptive_height:
            pos_y = im.shape[1]
        else:
            pos_y += receptive_height
            if pos_y > im.shape[1] - receptive_height:
                pos_y = im.shape[1] - receptive_height
    out = np.moveaxis(out, 0, 2) * 255.
    out[out < 0] = 0.
    out[out > 255] = 255.
    out_im = Image.fromarray(out.squeeze(), mode='L')
    out_im.save("out.tiff")


