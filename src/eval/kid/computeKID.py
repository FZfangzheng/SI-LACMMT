from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import json

from src.eval.kid.models.inception import InceptionV3
from src.eval.kid.kid_score import calculate_kid_given_paths
from src.eval.kid.fid_score import calculate_fid_given_paths

if __name__=='__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--true', type=str, required=True,
    #                     help=('Path to the true images'))
    # parser.add_argument('--fake', type=str, nargs='+', required=True,
    #                     help=('Path to the generated images'))
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('-c', '--gpu', default='0', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument('--model', default='inception', type=str,
                        help='inception or lenet')
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # paths = [args.true] + args.fake

    # truepaths=[r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.1-HN14-jointDL1\9.1.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.1-HN14-jointDL1\9.1.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.1-HN14-jointDL1\9.1.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.2-HN14-p2pHDmodel\9.2.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.2-HN14-p2pHDmodel\9.2.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.2-HN14-p2pHDmodel\9.2.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\real_result',
    #
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.1-p2pdata-jointDL1\10.1.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.1-p2pdata-jointDL1\10.1.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.1-p2pdata-jointDL1\10.1.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.2-p2pdata-p2pHDmodel\10.2.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.2-p2pdata-p2pHDmodel\10.2.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.2-p2pdata-p2pHDmodel\10.2.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.0\real_result',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.0\real_result',
    #            ]
    # fakepaths=[r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.1\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.2\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.3\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.1-HN14-jointDL1\9.1.0\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.1-HN14-jointDL1\9.1.1\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.1-HN14-jointDL1\9.1.2\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.2-HN14-p2pHDmodel\9.2.0\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.2-HN14-p2pHDmodel\9.2.1\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.2-HN14-p2pHDmodel\9.2.2\fake_result',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.0\seg_result_gray_repaint',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.1\seg_result_gray_repaint',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.2\seg_result_gray_repaint',
    #            r'D:\map_translate\看看效果\0731\9.0-HN14-connect_featuremap\9.0.3\seg_result_gray_repaint',
    #
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.0\fake_result',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.1\fake_result',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.2\fake_result',
    #            r'D:\map_translate\看看效果\0731\10.1-p2pdata-jointDL1\10.1.0\fake_result',
    #            r'D:\map_translate\看看效果\0731\10.1-p2pdata-jointDL1\10.1.1\fake_result',
    #            r'D:\map_translate\看看效果\0731\10.1-p2pdata-jointDL1\10.1.2\fake_result',
    #            r'D:\map_translate\看看效果\0731\10.2-p2pdata-p2pHDmodel\10.2.0\fake_result',
    #            r'D:\map_translate\看看效果\0731\10.2-p2pdata-p2pHDmodel\10.2.1\fake_result',
    #            r'D:\map_translate\看看效果\0731\10.2-p2pdata-p2pHDmodel\10.2.2\fake_result',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.0\seg_result_gray_repaint',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.1\seg_result_gray_repaint',
    #            r'D:\map_translate\看看效果\0731\10.0-p2pdata-connect_featuremap\10.0.2\seg_result_gray_repaint',
    #            ]

    truepaths = [r'D:\map_translate\看看效果\0829\9.5\9.5.0\real_result',
                 r'D:\map_translate\看看效果\0829\9.5\9.5.0\real_result',
                 r'D:\map_translate\看看效果\0829\9.5\9.5.0\real_result',

                 r'D:\map_translate\看看效果\0829\9.6\9.6.0\real_result',
                 r'D:\map_translate\看看效果\0829\9.6\9.6.0\real_result',
                 r'D:\map_translate\看看效果\0829\9.6\9.6.0\real_result',
                 ]
    fakepaths = [r'D:\map_translate\看看效果\0829\9.5\9.5.0\fake_result',
                 r'D:\map_translate\看看效果\0829\9.5\9.5.1\fake_result',
                 r'D:\map_translate\看看效果\0829\9.5\9.5.2\fake_result',

                 r'D:\map_translate\看看效果\0829\9.6\9.6.0\fake_result',
                 r'D:\map_translate\看看效果\0829\9.6\9.6.1\fake_result',
                 r'D:\map_translate\看看效果\0829\9.6\9.6.2\fake_result',
                 ]
    assert len(truepaths)==len(fakepaths)
    targetfile_kid='./kid.json'
    targetfile_fid='./fid.json'

    for i in range(len(truepaths)):
        paths=[truepaths[i]]+[fakepaths[i]]
        results = calculate_kid_given_paths(paths, args.batch_size, args.gpu != '', args.dims, model_type=args.model)
        with open(targetfile_kid, 'a') as f:
            json.dump(results, f)
            f.write('\n')
        for p, m, s in results:
            print('KID (%s): %.3f (%.3f)' % (p, m, s))

        paths=[truepaths[i]]+[fakepaths[i]]
        results = calculate_fid_given_paths(paths, args.batch_size, args.gpu != '', args.dims, model_type=args.model)
        with open(targetfile_fid, 'a') as f:
            json.dump(results, f)
            f.write('\n')
        for p, m, s in results:
            print('KID (%s): %.3f (%.3f)' % (p, m, s))

        print(f'No.{i} folder')
