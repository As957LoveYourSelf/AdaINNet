import os
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
from adainnet import AdainNet

test_content_images_path = "./images/test/content"
test_style_images_path = "./images/test/style"

test_content_images = []
for g in os.listdir(test_content_images_path):
    test_content_images.append(os.path.join(test_content_images_path, g))
# print(test_content_images)

test_style_images = []
for g in os.listdir(test_style_images_path):
    test_style_images.append(os.path.join(test_style_images_path, g))


# print(test_style_images)


def generate_images(content_path, style_path, alpha=1.0):
    re_train_model_path = "result_style_weight_10/model_state/20_epoch.pth"
    model = AdainNet().cuda()
    pre = torch.load(re_train_model_path)
    model.load_state_dict(pre)

    c_ = Image.open(content_path)
    s_ = Image.open(style_path)
    tran = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
         ]
    )

    c_ = tran(c_)
    c_ = Variable(torch.unsqueeze(c_, dim=0).float(), requires_grad=False).cuda()
    print(c_.shape)
    s_ = tran(s_)
    s_ = Variable(torch.unsqueeze(s_, dim=0).float(), requires_grad=False).cuda()
    print(s_.shape)
    with torch.no_grad():
        result_ = model.generator(c_, s_, alpha=alpha)

    return result_


c = test_content_images[6]
c_name = os.path.basename(c).split('.')[0]
s = test_style_images[9]
s_name = os.path.basename(s).split('.')[0]
c_s_zips = zip(test_content_images, test_style_images)
image_save_path = "result_style_weight_10/test_result"
if not os.path.exists(image_save_path):
    os.mkdir(image_save_path)
alpha_ = 1.0

# save one image
# result = generate_images(c, s, alpha_)
# save_image(result_style_weight_5, f"{image_save_path + '/' + c_name + '_' + s_name + '.jpg'}")
# save_image(result, f"{image_save_path + '/' + c_name + '_' + s_name + '_'+str(alpha_)+'.jpg'}")
# print("saved")

# save images
for c in test_content_images:
    for s in test_style_images:
        try:
            c_name = os.path.basename(c).split(".")[0]
            s_name = os.path.basename(s).split(".")[0]
            image_save_path = "./result_style_weight_10/test_result"
            if not os.path.exists(image_save_path):
                os.mkdir(image_save_path)
            result = generate_images(c, s, alpha_)
            save_image(result, f"{image_save_path + '/' + c_name + '_' + s_name + '_'+str(alpha_)+'.jpg'}")
            print("saved")
        except:
            continue
