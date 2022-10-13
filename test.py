import os
import time
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from models import model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False

TEST_SAMPLES = './test_samples'
TEST_RESULTS = './test_results'

def save_images(images, name):
    filename = TEST_RESULTS + '/' + name
    torchvision.utils.save_image(images, filename)

def main():
    lnnet = model(num_resblocks=[6, 6, 6, 6], input_channels=[3, 3, 3, 3]).to(device)
    lnnet.load_state_dict(torch.load('./ckpts/net_c8_n6.pth'))
    print('load deblurnet success')

    if os.path.exists(TEST_RESULTS) == False:
        os.mkdir(TEST_RESULTS)

    test_time = 0.0
    iteration = 1.0
    for images_name in os.listdir(TEST_SAMPLES):
        with torch.no_grad():
            input_image = transforms.ToTensor()(Image.open(TEST_SAMPLES + '/' + images_name).convert('RGB'))
            input_image = Variable(input_image-0.5).unsqueeze(0).to(device)

            torch.cuda.synchronize()
            start = time.time()
            d4 = lnnet(input_image)
            torch.cuda.synchronize()
            stop = time.time()
            test_time += stop - start
            save_images(d4+0.5, images_name)
            print('RunTime:%.4f' % (stop - start), 'Average Runtime:%.4f' % (test_time / iteration))
            iteration += 1

if __name__ == '__main__':
    main()