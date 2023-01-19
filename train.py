import random
from apex import amp
import numpy as np
from torchvision import transforms
import os, torch
import argparse
from Network import CDERNet
from PIL import Image
from loss import KL_loss, L1_loss, ratio_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_train', type=str, default='datasets/DEFE/cropped_data/train/', help='My trainset path.')
    parser.add_argument('--data_path_test', type=str, default='datasets/DEFE/cropped_data/test/', help='My testset path.')
    parser.add_argument('--class_num', type=int, default=3, help='Class numbers.')
    parser.add_argument('--embedding_size', type=int, default=256, help='Embedding size.')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model.')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=16, help='Batch size for validation.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 0)')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--alpha', type=float, default=0.7, help='alpha.')
    parser.add_argument('--save_path', type=str, default='model/', help='Saved model path.')
    return parser.parse_args()


transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

args = parse_args()
pic_list_train = os.listdir(args.data_path_train)
random.shuffle(pic_list_train)
pic_list_test = os.listdir(args.data_path_test)
random.shuffle(pic_list_test)


class mytrainset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        img_pil = Image.open(os.path.join(args.data_path_train, pic_list_train[index]))
        img_tensor = transform_train(img_pil)
        target = int(pic_list_train[index][0])
        return img_tensor, target

    def __len__(self):
        return len(pic_list_train)

class mytestset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        img_pil = Image.open(os.path.join(args.data_path_test, pic_list_test[index]))
        img_tensor = transform_test(img_pil)
        target = int(pic_list_test[index][0])
        return img_tensor, target

    def __len__(self):
        return len(pic_list_test)


train_dataset = mytrainset()
print('Train set size:', train_dataset.__len__())
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=args.workers,
                                           shuffle=True,
                                           pin_memory=True)

val_dataset = mytestset()
val_num = val_dataset.__len__()
print('Validation set size:', val_num)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.val_batch_size,
                                           num_workers=args.workers,
                                           shuffle=False,
                                           pin_memory=True)

def DRDM_loss(outputs, targets):
    DRDM_loss = KL_loss(outputs, targets)
    # DRDM_loss = L1_loss(outputs, targets)
    # DRDM_loss = ratio_loss(outputs, targets)
    return DRDM_loss

model = CDERNet(embedding_size=args.embedding_size, class_num=args.class_num, pretrained=args.pretrained)
if args.checkpoint:
    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
params = model.parameters()
print(args.alpha)
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-4)
else:
    raise ValueError("Optimizer not supported.")
print(optimizer)

model = model.cuda()
model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
CE_criterion = torch.nn.CrossEntropyLoss()

def run_training():
    args = parse_args()
    
    best_acc = 0
    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        train_drdmloss = 0.0
        train_celoss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        confusionMatrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for batch_i, (imgs, targets) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            embeddings, outputs = model(imgs)
            targets = targets.cuda()
            alpha = args.alpha
            drdm_loss = DRDM_loss(embeddings, targets)
            CE_loss = CE_criterion(outputs, targets)
            loss = (1 - alpha) * CE_loss + alpha * drdm_loss
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            
            train_loss += loss
            train_drdmloss += drdm_loss
            train_celoss += CE_loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            for j in range (predicts.size()[0]):
                confusionMatrix[targets[j].item()][predicts[j].item()] += 1

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss/iter_cnt
        train_drdmloss = train_drdmloss/iter_cnt
        train_celoss = train_celoss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f.  Total Loss: %.3f (Loss1: %.3f Loss2: %.3f) LR: %.6f' %
              (i, train_acc, train_loss, train_drdmloss, train_celoss, optimizer.param_groups[0]["lr"]))
        print(confusionMatrix)
        scheduler.step()

        with torch.no_grad():
            val_loss = 0.0
            val_drdmloss = 0.0
            val_celoss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            confusionMatrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for batch_i, (imgs, targets) in enumerate(val_loader):
                embeddings, outputs = model(imgs.cuda())
                targets = targets.cuda()
                alpha = args.alpha
                drdm_loss = DRDM_loss(embeddings, targets)
                CE_loss = CE_criterion(outputs, targets)
                loss = (1 - alpha) * CE_loss + alpha * drdm_loss
                val_loss += loss
                val_drdmloss += drdm_loss
                val_celoss += CE_loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                for j in range (predicts.size()[0]):
                    confusionMatrix[targets[j].item()][predicts[j].item()] += 1
                bingo_cnt += correct_or_not.sum().cpu()
                
            val_loss = val_loss/iter_cnt
            val_drdmloss = val_drdmloss/iter_cnt
            val_celoss = val_celoss/iter_cnt
            val_acc = bingo_cnt.float()/float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            print("[Epoch %d] Validation accuracy:%.4f. Total Loss: %.3f (Loss1: %.3f Loss2: %.3f)" % 
                    (i, val_acc, val_loss, val_drdmloss, val_celoss))
            print(confusionMatrix)

            if val_acc > best_acc:
                best_acc = val_acc
                print("best_acc:" + str(best_acc))
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join(args.save_path, "lambda" + str(args.alpha) + "_acc" + str(val_acc) + ".pth"))
                print('Model saved.')      
    print(best_acc)

run_training()
