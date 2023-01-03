import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import time

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test

    # Set eval mode. This only affects layers like batch norm and drop out. 
    if opt.eval:
        model.eval()

    count_time=0
    loop_time=0
    for i, data in enumerate(dataset):
        # if i >= opt.how_many:
        #     break
        start = time.clock()
        model.set_input(data)

        model.test()
        # 获取结束时间
        loop_time = loop_time + 1
        end = time.clock()
        # 计算运行时间
        runTime = end - start
        # print("运行时间：", runTime, "秒")
        count_time = count_time + runTime
        if loop_time%100 == 0 and loop_time!=0:
            print("运行时间100：", count_time, "秒")
            count_time=0
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
