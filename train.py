import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
import os

import function
from dataset import PreprocessDataSet
from adainnet import AdainNet
from torchvision.utils import save_image
from tqdm import tqdm

BATCH_SIZE = 8
EPOCH = 20
LR = 4e-5
snapshot_interval = (60000 // BATCH_SIZE) // 2
print(f"snapshot_interval: {snapshot_interval}")
save_dir = "result_style_weight_10"
train_content_dir = "./images/train/content_images"
train_style_dir = "./images/train/style_images"
test_content_dir = "./images/test/content"
test_style_dir = "./images/test/style"


def main():
    # create directory to save
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    loss_dir = f'{save_dir}/loss'
    model_state_dir = f'{save_dir}/model_state'
    image_dir = f'{save_dir}/image'

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
        os.mkdir(model_state_dir)
        os.mkdir(image_dir)

    # set device on GPU if available, else CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    print(f'# Minibatch-size: {BATCH_SIZE}')
    print(f'# epoch: {EPOCH}')

    # prepare dataset and dataLoader
    train_dataset = PreprocessDataSet(train_content_dir, train_style_dir)
    test_dataset = PreprocessDataSet(test_content_dir, test_style_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_iter = list(test_loader)
    # print(len(test_iter))

    # set model and optimizer
    re_train_model_path = "result_style_weight_10/model_state/12_epoch.pth"
    model = AdainNet().to(device)
    pre = torch.load(re_train_model_path)
    model.load_state_dict(pre)
    optimizer = Adam(model.parameters(), lr=LR)

    # start training
    # loss_list = []
    for e in range(13, EPOCH + 1):
        print(f'Start {e} epoch')
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        for i, (content, style) in tqdm(enumerate(train_loader, 1)):
            content = content.to(device)
            style = style.to(device)
            loss = model(content, style)
            # loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'[{e}/total {EPOCH} epoch],[{i} /'
                  f'total {round(iters / BATCH_SIZE)} iteration]: {loss.item()}')

            if i % snapshot_interval == 0:
                content, style = test_iter[e % 2]
                content = content.to(device)
                style = style.to(device)
                with torch.no_grad():
                    out = model.generator(content, style)
                content = function.denorm(content, device)
                style = function.denorm(style, device)
                out = function.denorm(out, device)
                res = torch.cat([content, style, out], dim=0)
                res = res.to('cpu')
                save_image(res, f'{image_dir}/{e}_epoch_{i}_iteration.png', nrow=BATCH_SIZE)
        torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
    # plt.plot(range(len(loss_list)), loss_list)
    # plt.xlabel('iteration')
    # plt.ylabel('loss')
    # plt.title('train loss')
    # plt.savefig(f'{loss_dir}/train_loss.png')
    # with open(f'{loss_dir}/loss_log2.txt', 'w') as f:
    #     for l in loss_list:
    #         f.write(f'{l}\n')
    # print(f'Loss saved in {loss_dir}')


if __name__ == '__main__':
    main()
