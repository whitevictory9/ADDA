import argparse
import numpy as np
import torch
from torch.autograd import Variable
from utils import IMAGENET, MyDataset

# TODO: change the below to point to the ImageNet validation set
IMAGENET_PATH = ""
if IMAGENET_PATH == "":
    raise ValueError("Please fill out the path to ImageNet")


def norm(t):
    assert len(t.shape) == 4
    norm_vec = torch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec


def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + torch.clamp(new_x - orig, -eps, eps)
    return proj

def l2_step(x, g, lr):
    return x + lr*g/norm(g)

def linf_step(x, g, lr):
    return x + lr*torch.sign(g)

def l2_step_t(x, g, lr):
    return x - lr*g/norm(g)

def linf_step_t(x, g, lr):
    return x - lr*torch.sign(g)

def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target

def one_hot(index, total):
    arr = torch.zeros((total))
    arr[index] = 1
    return arr

class ADD(object):
    def __init__(self, cnn_model, sur_model):
        self.cnn_model = cnn_model
        self.sur_model = sur_model

    def adv_grad(self, x, direction):
        x = Variable(x.data, requires_grad=True)
        drawModelIndex = np.random.randint(len(self.sur_model))
        surrogate_model = self.sur_model[drawModelIndex]
        loss = (surrogate_model.predict_scores(x) * direction).sum()
        loss.backward()
        return x.grad.data

    def untargeted_attack(self, x_in, y, eps, norm_type, max_query, lr, num_classes):
        x_adv = x_in.clone()
        att_query = 0
        oso_step = l2_step_t if norm_type == 'l2' else linf_step_t
        proj_maker = l2_proj if norm_type == 'l2' else linf_proj
        proj_step = proj_maker(x_in, eps)
        direction = one_hot(y, num_classes).to(device)

        while att_query < max_query:
            if self.cnn_model.predict_label(x_adv) == y:
                att_query += 1
                grad = self.adv_grad(x_adv, direction)
                new_im = oso_step(x_adv, grad, lr)
                image = proj_step(new_im)
                x_adv = torch.clamp(image, 0, 1)
            else:
                att_query += 1
                break
        return x_adv, att_query

    def targeted_attack(self, x_in, target_class, odi_step, eps, norm_type, max_query, num_classes):
        x_adv = x_in.clone()
        att_query = 0
        ods_step = l2_step if norm_type == 'l2' else linf_step
        proj_maker = l2_proj if norm_type == 'l2' else linf_proj
        proj_step = proj_maker(x_in, eps)
        direction = one_hot(target_class, num_classes).to(device)

        while att_query < max_query:
            if self.cnn_model.predict_label(x_adv) != target_class:
                att_query += 1
                grad = self.adv_grad(x_adv, direction)
                new_im = ods_step(x_adv, grad, odi_step)
                image = proj_step(new_im)
                x_adv = torch.clamp(image, 0, 1)
            else:
                att_query += 1
                break
        return x_adv, att_query


def main():

    cnn_model = IMAGENET(args.archTarget)
    num_classes = 1000
    attr_list = ['resnet152', 'resnext101_32x8d', 'vgg13', 'wide_resnet101_2'] # you can set other surrogate models.
    surrogate_model_list = []
    for i in range(len(attr_list)):
        surrogate_model_list.append(IMAGENET(attr_list[i]))

    if args.archTarget == 'inception_v3':
        test_dataset = MyDataset(299, 299)
    else:
        test_dataset = MyDataset()

    samples = list(range(1000))
    samples = samples[args.start: args.start + args.num_attacks]
    print("Length of sample_set: ", len(samples))

    #loop over images, attacking each one if it is initially correctly classified
    for idx in samples[:args.num_attacks]:
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        print(f"Image {idx:d}   Original label: {label:d}")
        predicted_label = cnn_model.predict_label(image)

        # ignore incorrectly classified images
        if label == predicted_label:
            if args.target_class < 0:
                add = ADD(cnn_model, surrogate_model_list)

                x_adv, att_query = add.untargeted_attack(image, predicted_label, args.epsilon, args.norm_type, args.max_query, args.lr, num_classes)
                adv_label = cnn_model.predict_label(x_adv)
                if adv_label != label:
                    print('adv_label{}, query {}'.format(adv_label, att_query))

                else:
                    print('Untargeted attack fail!')

            else:
                target_label = pseudorandom_target(idx, num_classes, label)
                print('choose pseudorandom target class: %d' % target_label)
                add = ADD(cnn_model, surrogate_model_list)
                x_adv, att_query = add.targeted_attack(image, target_label, args.lr, args.epsilon, args.norm_type, args.max_query, num_classes)
                print('att_query', att_query)
                adv_label = cnn_model.predict_label(x_adv)
                if adv_label == target_label:
                    print('adv_label{}, query {}'.format(adv_label, att_query))
                else:
                    print('Targeted attack fail!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Imagenet PGD Attack Evaluation with Output Diversified Initialization')
    parser.add_argument('--archTarget', type=str, help='model to be attacked, [resnet50, vgg16_bn, or densenet121]')
    parser.add_argument('--dset', type=str, default='imagenet', help='Dataset to be used')
    parser.add_argument('--norm_type', type=str, help='l_p norm type, could be l2 or linf')
    parser.add_argument('--epsilon', type=float, help='allowed l_p perturbation size')
    parser.add_argument('--num_attacks', type=int, default=1000, help='number of images to attack')
    parser.add_argument('--start', type=int, default=0, help='index of first image to attack')
    parser.add_argument('--max_query', type=int, help='maximum allowed queries for each image')
    parser.add_argument('--target_class', type=int, default=-1, help='negative => untargeted')
    parser.add_argument('--lr', type=float, help='pgd step size')
    args = parser.parse_args()
    print(args)

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # do the business
    main()

