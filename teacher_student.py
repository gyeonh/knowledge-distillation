
import argparse
import os
import json
import torch 
import random 
import numpy as np
import torchvision
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

# import albumentations as A
# from albumentations.pytorch import ToTensorV2


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False    


def getdata(root_dir, batch_size, resize):
    mean = [0.4919, 0.4826, 0.4470]
    stev = [0.2408, 0.2373, 0.2559]
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize, resize)), 
        torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(mean, stev)
    ])
    test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, stev)
        ])
    train_dataset = datasets.CIFAR10(root = f"{root_dir}/train", train = True, download=False, transform = train_transforms)
    test_dataset = datasets.CIFAR10(root = f"{root_dir}/test", train = False, download=False, transform = test_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return train_dataloader, test_dataloader


def ResNet(model_name):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained = True)
    elif model_name == 'resnet152':
        model = torchvision.models.resnet152(pretrained = True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    torch.nn.init.xavier_uniform_(model.fc.weight)
    return model 

def Efficientnet(model_name):
    if model_name == 'efficientnet_b1':
        model = torchvision.models.efficientnet_b1(pretrained=True)
    elif model_name == 'efficientnet_b3':
        model = torchvision.models.efficientnet_b3(pretrained = True)
    elif model_name == 'efficientnet_b5':
        model = torchvision.models.efficientnet_b5(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    torch.nn.init.xavier_uniform_(model.classifier[1].weight)
    return model 


def get_model(model_name):
    if 'resnet' in model_name:
        return ResNet(model_name)
    elif 'efficientnet' in model_name:
        return Efficientnet(model_name)

def json_save(path, result):
    with open(path, "w") as f:
        json.dump(result, f, indent="\t", default=str)


def eval(model):
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        with tqdm(total=len(test_dataloader), desc="test") as pbar:
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs)
                test_acc += (torch.argmax(preds, dim=-1) == labels).sum()
                pbar.update(1)
    test_acc = (test_acc/len(test_dataloader.dataset))*100
    return test_acc 

def teacher_train():    
    results = {'train_loss':[], 'test_acc':[]}
    print(f".. teacher train start {args.epochs}")
    for e in range(1, args.epochs+1):
        teacher_model.train()
        epoch_train_loss = 0.0
        with tqdm(total = len(train_dataloader), desc =f"{e}/{args.epochs}") as pbar:
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = teacher_model(inputs)
                loss = ce(preds, labels)
                teacher_optimizer.zero_grad()
                loss.backward()
                teacher_optimizer.step()
                epoch_train_loss += loss.item()
                pbar.update(1)
        epoch_train_loss /= len(train_dataloader)
        results['train_loss'].append(epoch_train_loss)

        test_acc = eval(teacher_model)
        results['test_acc'].append(test_acc)
        
        if test_acc == max(results['test_acc']):
            if e == 20:
                # PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_teacher_best_model_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}_{test_acc}.pth'
                PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_teacher_best_model_ce_{test_acc}.pth'
                torch.save(teacher_model.state_dict(), PATH)
            # PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_teacher_best_model_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}.pth'
            PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_teacher_best_model_ce.pth'
            torch.save(teacher_model.state_dict(), PATH)
        
        print(f".. epoch {e} result: loss={epoch_train_loss:.5f}, accuracy={test_acc:.5f}")
    return results

def get_student_loss(t_output, s_output, labels):
    kld_loss = kld(F.softmax(s_output, dim=-1), F.log_softmax(t_output, dim=-1))
    hard_mse_loss = mse(s_output, t_output)
    soft_mse_loss = mse(F.softmax(s_output, dim=-1), F.log_softmax(t_output, dim=-1))
    ce_loss = ce(s_output, labels)
    loss = args.kld_lambda*kld_loss + args.hard_mse_lambda*hard_mse_loss + args.soft_mse_lambda*soft_mse_loss + args.ce_lambda*ce_loss
    return ce_loss

def student_train():    
    results = {'train_loss':[], 'test_acc':[]}
    print(f".. student train start {args.epochs}")
    for e in range(1, args.epochs+1):
        teacher_model.eval()
        student_model.train()
        epoch_train_loss = 0.0
        with tqdm(total = len(train_dataloader), desc =f"{e}/{args.epochs}") as pbar:
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                t_output = teacher_model(inputs)
                s_output = student_model(inputs)
                loss = get_student_loss(t_output, s_output, labels)
                student_optimizer.zero_grad()
                loss.backward()
                student_optimizer.step()
                epoch_train_loss += loss.item()
                pbar.update(1)
        epoch_train_loss /= len(train_dataloader)
        results['train_loss'].append(epoch_train_loss)
        test_acc = eval(student_model)
        results['test_acc'].append(test_acc)
        
        if test_acc == max(results['test_acc']):
            if e == 20:
                # PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_student_best_model_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}_{test_acc}.pth'
                PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_student_best_model_ce_{test_acc}.pth'
                torch.save(student_model.state_dict(), PATH)
            # PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_student_best_model_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}_{test_acc}.pth'
            PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_student_best_model_ce_{test_acc}.pth'
            torch.save(student_model.state_dict(), PATH)
        print(f".. epoch {e} result: loss={epoch_train_loss:.5f}, accuracy={test_acc:.5f}")
    return results

def join_train():
    results = {'teacher_loss':[], 'student_loss':[], 'teacher_acc':[], 'student_acc':[]}
    print(f".. teacher, student train start {args.epochs}")
    for e in range(1, args.epochs+1):
        teacher_model.train()
        student_model.train()
        epoch_t_loss = 0.0
        epoch_s_loss = 0.0 
        with tqdm(total = len(train_dataloader), desc =f"{e}/{args.epochs}") as pbar:
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                #teacher train
                t_output = teacher_model(inputs)
                t_loss = ce(t_output, labels)
                teacher_optimizer.zero_grad()
                t_loss.backward()
                teacher_optimizer.step()
                epoch_t_loss += t_loss.item()

                #student train
                s_output = student_model(inputs)
                t_output = t_output.detach()
                s_loss = get_student_loss(t_output, s_output, labels)
                student_optimizer.zero_grad()
                s_loss.backward()
                student_optimizer.step()
                epoch_s_loss += s_loss.item()
                pbar.update(1)
                
        epoch_t_loss /= len(train_dataloader)
        epoch_s_loss /= len(train_dataloader)
        results['teacher_loss'].append(epoch_t_loss)
        results['student_loss'].append(epoch_s_loss)

        teacher_test_acc = eval(teacher_model)
        student_test_acc = eval(student_model)
        results['teacher_acc'].append(teacher_test_acc)
        results['student_acc'].append(student_test_acc)
        
        if teacher_test_acc == max(results['teacher_acc']):
            if e == 20:
                # PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_teacher_best_model_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}_{teacher_test_acc}.pth'
                PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_teacher_best_model_ce_{teacher_test_acc}.pth'
                torch.save(teacher_model.state_dict(), PATH)
            # PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_teacher_best_model_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}.pth'
            PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_teacher_best_model_ce.pth'
            torch.save(teacher_model.state_dict(), PATH)
            
        if student_test_acc == max(results['student_acc']):
            if e == 20:
                # PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_student_best_model_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}_{student_test_acc}.pth'
                PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_student_best_model_ce_{student_test_acc}.pth'
                torch.save(student_model.state_dict(), PATH)
            # PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_student_best_model_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}.pth'
            PATH = f'/NasData/home/hjy/comvi_pjt/results/{args.experiment}/model/{args.train_mode}_student_best_model_ce.pth'
            torch.save(student_model.state_dict(), PATH)
            
        print(f".. epoch {e} teacher result: loss={epoch_t_loss:.5f}, accuracy={teacher_test_acc:.5f}")
        print(f".. epoch {e} student result: loss={epoch_s_loss:.5f}, accuracy={student_test_acc:.5f}")
    return results    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help= 'batch_size')
    parser.add_argument('--resize', type=int, default=224, help= 'resize')
    parser.add_argument('--student_lr', type=float, default=1e-5, help= 'student_lr')
    parser.add_argument('--teacher_lr', type=float, default=1e-5, help= 'teacher_lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help= 'weight_decay')
    parser.add_argument('--epochs', type=int, default=20, help= 'epochs')
    parser.add_argument('--gpus', type=str, default='2, 3', help= 'gpus')
    parser.add_argument('--teacher', type=str, default='resnet152', help= 'teacher')
    parser.add_argument('--student', type=str, default='resnet18', help= 'student')
    parser.add_argument('--ckpt', type=str, default='t=resnet152', help= 'ckpt')
    parser.add_argument('--root_dir', type=str, default='./data', help= 'data_dir')
    parser.add_argument('--train_mode', type=str, default= 'join', help= 'join: 1 steps-> teacher & student train or each: after teacher train, student train')
    parser.add_argument('--seed', type=int, default= '1234', help= 'seed')
    parser.add_argument('--multi_gpus', type=int, default=1, help= 'multi_gpus')
    parser.add_argument('--kld_lambda', type=float, default=0.2, help= 'student: 1.0~0.0')
    parser.add_argument('--hard_mse_lambda', type=float, default=0.1, help= 'student: 1.0~0.0')
    parser.add_argument('--soft_mse_lambda', type=float, default=0.1, help= 'student: 1.0~0.0')
    parser.add_argument('--ce_lambda', type=float, default=0.6, help='student: 1.0~0.0')
    parser.add_argument('--experiment', type=str, default='experiment2', help='nth experiment')
    
    args = parser.parse_args()
    print(f".. seed {args.seed} setting")
    set_seed(args.seed)

    save_path = '/NasData/home/hjy/comvi_pjt/results/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #     os.makedirs(save_path+'/model')
    print(f'.. your save path: {save_path}')

    
    print(f".. gpus {args.gpus} setting")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device} /// gpus: {args.gpus} /// multi:{args.multi_gpus}')


    print(f".. load dataset")
    train_dataloader, test_dataloader = getdata(args.root_dir, args.batch_size, args.resize)
    

    print(f".. load model teacher={args.teacher}, student={args.student}")
    teacher_model = get_model(args.teacher).to(device)
    student_model = get_model(args.student).to(device)

    if args.multi_gpus:
        student_model = nn.DataParallel(student_model).to(device)
        teacher_model = nn.DataParallel(teacher_model).to(device)
    else:
        student_model = student_model.to(device)
        teacher_model = teacher_model.to(device)


    teacher_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=args.teacher_lr, weight_decay=0.01)
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.student_lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss(reduction="mean") #(input, target)
    kld = nn.KLDivLoss(reduction="batchmean") #(log softmax input, softmax target)


    # 모든 epoch에 대한 teacher/student acc 및 loss 저장된 json 파일
    if args.train_mode == "each":
        teacher_result = teacher_train()
        student_result = student_train()
        # json_save(save_path + f'{args.experiment}/{args.train_mode}/teacher_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}.json', teacher_result)
        json_save(save_path + f'{args.experiment}/{args.train_mode}/teacher_ce.json', teacher_result)
        # json_save(save_path + f'{args.experiment}/{args.train_mode}/student_{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}.json', student_result)
        json_save(save_path + f'{args.experiment}/{args.train_mode}/student_ce.json', student_result)
            
    elif args.train_mode == "join":
        tot_result = join_train()
        # json_save(save_path + f'{args.experiment}/{args.train_mode}/{args.kld_lambda}_{args.hard_mse_lambda}_{args.soft_mse_lambda}_{args.ce_lambda}.json', tot_result)
        json_save(save_path + f'{args.experiment}/{args.train_mode}/ce.json', tot_result)
