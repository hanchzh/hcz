from pyexpat import model
from models.test import test_img
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_train_test
import torch
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
args = args_parser()


class Fed:
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device('cuda:{}'.format(
            self.args.gpu) if torch.cuda.is_available() and self.args.gpu != -1 else 'cpu')
        self.num_users = args.num_users

    def prepare(self):
        self.load_data_fl()
        self.build_model()
    def load_data_sfl(self):
        # load dataset and split users
        if self.args.dataset == 'mnist':
            trans_mnist = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST(
                './data/mnist/', train=True, download=False, transform=trans_mnist)
            dataset_test = datasets.MNIST(
                './data/mnist/', train=False, download=False, transform=trans_mnist)
            # sample users
            if self.args.iid:
                # 如果是iid 的数据集,key值是用户id，value是用户拥有的图片
                dict_users = mnist_iid(dataset_train, self.args.num_users)
            else:
                # 如果是no-iid 的数据集
                dict_users, dict_users_test = mnist_noniid_train_test(dataset_train, dataset_test, self.args.num_users)
        elif self.args.dataset == 'cifar':
            # cifar 数据集
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10(
                './data/cifar', train=True, download=False, transform=trans_cifar)
            dataset_test = datasets.CIFAR10(
                './data/cifar', train=False, download=False, transform=trans_cifar)
            if self.args.iid:
                dict_users = cifar_iid(dataset_train, self.args.num_users)
            else:
                exit('Error: only consider IID setting in CIFAR10')
        else:
            exit('Error: unrecognized dataset')
        img_size = dataset_train[0][0].shape
        '''我们通过定义不同的数据划分方式将数据分为 iid 和 non-iid 两种，用来模拟测试 FedAvg 在不同场景下的性能。返回的是一个字典类型 dict_users，key值是用户 id，values是用户拥有的图片id。(具体实现方式可以自行研究代码)
        '''
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.dict_users_test = dict_users_test
        self.img_size = img_size
        return dataset_train, dataset_test, dict_users, dict_users_test
    def load_data_fl(self, set = 0):
        # load dataset and split users
        dataset_trains = []
        dataset_tests = []


        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307+set,), (0.3081+set,))])
        dataset_train = datasets.MNIST(
            './data/mnist/', train=True, download=False, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            './data/mnist/', train=False, download=False, transform=trans_mnist)
        # dataset_trains.append(dataset_test)
        # dataset_tests.append(dataset_train)
        # sample users

        # key值是用户id，value是用户拥有的图片
        dict_users = mnist_iid(dataset_train, self.args.num_users)

        img_size = dataset_train[0][0].shape

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.img_size = img_size
        return dataset_train, dataset_test,dict_users

    def build_model(self):
        # build model
        if self.args.model == 'cnn' and self.args.dataset == 'cifar':
            net_glob = CNNCifar(args=self.args).to(self.args.device)
        elif self.args.model == 'cnn' and self.args.dataset == 'mnist':
            net_glob = CNNMnist(args=self.args).to(self.args.device)
        elif self.args.model == 'mlp':
            len_in = 1
            for x in self.img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200,
                           dim_out=self.args.num_classes).to(self.args.device)
        else:
            exit('Error: unrecognized model')
        print(net_glob)
        net_glob.train()
        self.net_glob = net_glob
        return net_glob


    def plot(self, curve):
        # plot loss curve
        plt.figure()
        plt.plot(range(len(curve)), curve)
        plt.ylabel('train_loss')
        plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(self.args.dataset,
                                                               self.args.model, self.args.epochs, self.args.frac,
                                                               self.args.iid))

    def plot_acc(self, curve):
        # plot loss curve
        plt.figure()
        plt.plot(range(len(curve)), curve)
        plt.ylabel('acc_test')
        plt.savefig('./save/fed_acc_{}_{}_{}_C{}_iid{}.png'.format(self.args.dataset,
                                                                   self.args.model, self.args.epochs, self.args.frac,
                                                                   self.args.iid))

    def testing(self, net_glob,dataset_train,dataset_test,):
        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        return acc_train, acc_test
    def dis_testing(self, net_glob):
        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, self.dict_users[0], args)
        acc_test, loss_test = test_img(net_glob, self.dict_users[0], args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        return acc_train, acc_test

def dispatch(w, client_a, client_b):
    # 将build的模型权重复制到全局模型
    # copy weight to net_glob
    # 复制到总的net，每个用户进行更新
    w_overlap = copy.deepcopy(w[0])
    for k in w_overlap.keys():
        w_overlap[k] = (len(client_a) * w[0][k] + len(client_b) * (w[1][k]))
        # 写法正确，接下来要考虑如何判断位于overlap区域的问题
        # i是list的编号，K是layer的层数
        # 每一层都要做div，加一起以后除以总数就行
        w_overlap[k] = torch.div(w_overlap[k], (len(client_a) + len(client_b)))
    return w_overlap


def fedAvg(w):
    w_avg = copy.deepcopy(w[0])
    key = w_avg.keys()
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
def sfed(w, w_users):
    for w_user in w_users:
        for k in list(w_user.keys())[:0]:
            w_user[k] = w[k]
    return w_users
def sfedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in list(w_avg.keys())[:0]:
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def fed_weight(w):
    # 如果位于overlap，权重参数会大一些,这里后面再做修改
    alfa_u = 1
    alfa_v = 1.5
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test():
    fed_a = Fed(args)
    fed_b = Fed(args)
    dataset_train_a, dict_users_a = fed_a.load_data_fl()
    dataset_train_b, dict_users_b = fed_b.load_data_fl()
    net_glob_a = fed_a.build_model()
    net_glob_b = fed_b.build_model()
    net_local = net_glob_a
    w_glob_a = net_glob_a.state_dict()
    w_glob_b = net_glob_b.state_dict()

    for idx in range(args.num_users):
        local_a = LocalUpdate(args=args, dataset=dataset_train_a, idxs=dict_users_a[idx])
        local_b = LocalUpdate(args=args, dataset=dataset_train_b, idxs=dict_users_b[idx])
        w_a, loss_a = local_a.train(net=copy.deepcopy(net_glob_a).to(args.device))
        w_b, loss_b = local_b.train(net=copy.deepcopy(net_glob_b).to(args.device))

    once_w_a = [w_a, w_b]
    test_avg = fedAvg(once_w_a)
    non_overlap_clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_overlap = dispatch(w_a, w_b, non_overlap_clients, non_overlap_clients)

    print("That is avg alg {}", test_avg)
    print("That is dispatch {}", test_overlap)


if __name__ == "__main__":
    fed_a = Fed(args)
    curve = [1,13,1.4,5.6,7.1]
    plt.ylabel('{%s}', list(dict(curve=curve).keys()[0]))
