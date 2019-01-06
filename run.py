import tensorflow as tf
import argparse
from Model import CRNN

parser = argparse.ArgumentParser(description='train or test the CRNN model')

parser.add_argument('--phase', dest='phase', default='train', help='train or test')
# parser.add_argument('--phase', dest='phase', default='test', help='train or test')
parser.add_argument('--bs', dest='batch_size', type=int, default=64,
                    help='size of a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=10000,
                    help='How many iteration in training')
parser.add_argument('--data_path', dest='dataset_path', default='G:\\图像文字识别\\Train_en_10000\\train_dataset.tfrecords',
                    help='where the image data is ')
# parser.add_argument('--data_path', dest='dataset_path', default='G:\\图像文字识别\\Train_en_10000\\test_dataset.tfrecords',
#                     help='where the image data is ')
parser.add_argument('--inti_lr', dest='init_learning_rate', type=float, default=0.001,
                    help='the initial learning rate when gradient')
parser.add_argument('--md', dest='max_width', type=int, default=1000,
                    help='the defined max width of the image')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./saver',
                    help='models are saved here')

args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(argv):
    crnn = CRNN(batch_size=args.batch_size,
                init_learning_rate=args.init_learning_rate,
                epochs=args.epoch,
                dataset_path=args.dataset_path,
                max_width=args.max_width,
                checkpoint_dir=args.checkpoint_dir
                )
    if args.phase is 'train':
        crnn.train()
    elif args.phase is 'test':
        crnn.test()

if __name__ == '__main__':
    tf.app.run()