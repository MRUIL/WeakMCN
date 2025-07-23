
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from io import BytesIO


class ProgressMeter(object):
    def __init__(self, version,num_epochs, num_batches, meters, prefix=""):
        self.fmtstr = self._get_epoch_batch_fmtstr(version,num_epochs, num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch, batch):
        entries = [self.prefix + self.fmtstr.format(epoch, batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_epoch_batch_fmtstr(self, version,num_epochs, num_batches):
        num_digits_epoch = len(str(num_epochs // 1))
        num_digits_batch = len(str(num_batches // 1))
        epoch_fmt = '{:' + str(num_digits_epoch) + 'd}'
        batch_fmt = '{:' + str(num_digits_batch) + 'd}'
        return '[' 'version: '+version+' '+ epoch_fmt + '/' + epoch_fmt.format(num_epochs) + ']' + '[' + batch_fmt + '/' + batch_fmt.format(
            num_batches) + ']'
#
#
# def log_scalar(callback, tag, value, step):
#     """Log a scalar variable.
#
#     Parameter
#     ----------
#     tag : basestring
#         Name of the scalar
#     value
#     step : int
#         training iteration
#     """
#     summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
#                                                  simple_value=value)])
#     callback.writer.add_summary(summary, step)
#
#
# def log_images(callback, tag, images, step):
#     """Logs a list of images."""
#
#     im_summaries = []
#     for nr, img in enumerate(images):
#         # Write the image to a string
#         s = BytesIO()
#         plt.imsave(s, img, format='png')
#
#         # Create an Image object
#         img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
#                                    height=img.shape[0],
#                                    width=img.shape[1])
#         # Create a Summary value
#         im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
#                                              image=img_sum))
#
#     # Create and write Summary
#     summary = tf.Summary(value=im_summaries)
#     callback.writer.add_summary(summary, step)
#
#
# def log_histogram(callback, tag, values, step, bins=1000):
#     """Logs the histogram of a list/vector of values."""
#     # Convert to a numpy array
#     values = np.array(values)
#
#     # Create histogram using numpy
#     counts, bin_edges = np.histogram(values, bins=bins)
#
#     # Fill fields of histogram proto
#     hist = tf.HistogramProto()
#     hist.min = float(np.min(values))
#     hist.max = float(np.max(values))
#     hist.num = int(np.prod(values.shape))
#     hist.sum = float(np.sum(values))
#     hist.sum_squares = float(np.sum(values ** 2))
#
#     # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
#     # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
#     # Thus, we drop the start of the first bin
#     bin_edges = bin_edges[1:]
#
#     # Add bin edges and counts
#     for edge in bin_edges:
#         hist.bucket_limit.append(edge)
#     for c in counts:
#         hist.bucket.append(c)
#
#     # Create and write Summary
#     summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
#     callback.writer.add_summary(summary, step)
#     callback.writer.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.avg_reduce = 0.

    def update(self, val, n=1):
        self.val = val
        if n==-1:
            self.sum=val
            self.count=1
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

    def update_reduce(self, val):
        self.avg_reduce = val

    def __str__(self):
        fmtstr = '{name} {avg_reduce' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

