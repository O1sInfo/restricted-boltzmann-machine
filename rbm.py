from time import time
import numpy as np
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import matplotlib.pyplot as plt


class RBM:
    def __init__(self, n_visible, n_hidden, cd_k=1, momentum=0.9,
        lr=0.01, weight_decay=0.0001, pretrained=None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if pretrained:
            weights = np.load(pretrained)
            self.w = torch.from_numpy(weights['weight']).to(self.device)
            self.w_m = torch.from_numpy(weights['weight_momentum']).to(self.device)
            self.vbias = torch.from_numpy(weights['vbias']).to(self.device)
            self.vbias_m = torch.from_numpy(weights['vbias_momentum']).to(self.device)
            self.hbias = torch.from_numpy(weights['hbias']).to(self.device)
            self.hbias_m = torch.from_numpy(weights['hbias_momentum']).to(self.device)
        else:
            self.w = (0.1 * torch.randn(n_visible, n_hidden)).to(self.device)
            self.w_m = torch.zeros(n_visible, n_hidden).to(self.device)
            self.hbias = torch.zeros(1, n_hidden).to(self.device)
            self.hbias_m = torch.zeros(1, n_hidden).to(self.device)
            self.vbias = torch.zeros(1, n_visible).to(self.device)
            self.vbias_m = torch.zeros(1, n_visible).to(self.device)
        self.cd_k = cd_k
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay


    def sample_h(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.w) + self.hbias)
        h = h_prob > torch.rand(h_prob.size()).to(self.device)
        h = h.type(torch.FloatTensor).to(self.device)
        return (h_prob, h)

    def sample_v(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.w.t()) + self.vbias)
        v = v_prob > torch.rand(v_prob.size()).to(device)
        v = v.type(torch.FloatTensor).to(self.device)
        return (v_prob, v)

    def constrastive_divergence(self, input):
        h_prob, h = self.sample_h(input)
        hidden = h_prob
        positive_gradient = torch.matmul(input.t(), h_prob)
        for k in range(self.cd_k):
            v_prob, v = self.sample_v(h)
            h_prob, h = self.sample_h(v_prob)
        negative_gradient = torch.matmul(v_prob.t(), h_prob)
        loss = positive_gradient - negative_gradient
        self.w_m = self.w_m * self.momentum + (1 - self.momentum) * loss
        self.vbias_m = self.vbias_m * self.momentum + \
            (1 - self.momentum) * torch.sum(input - v_prob, dim=0)
        self.hbias_m = self.hbias_m * self.momentum + \
            (1 - self.momentum) * torch.sum(hidden - h_prob, dim=0)
        batch_size = input.size(0)
        self.w += self.w_m * self.lr / batch_size
        self.w -= self.w * self.weight_decay
        self.vbias += self.vbias_m * self.lr / batch_size
        self.hbias += self.hbias_m * self.lr / batch_size
        # Compute reconstruction error
        error = torch.sum((input - v_prob)**2) / batch_size
        return error


def visualize(images):
    epochs = len(images)
    im_num = len(images[0])
    fig, axes = plt.subplots(epochs, im_num)
    for i, im_list in enumerate(images):
        for j, im in enumerate(im_list):
            axes[i][j].imshow(im, cmap=plt.cm.gray)
    for ax in axes.ravel():
        ax.set_axis_off()
    epochs_str = [str(i) for i in epochs_viz]
    epochs_str = '-'.join(epochs_str)
    filename = "rbm_{}_{}_e{}.png".format(rbms[-1].n_visible, rbms[-1].n_hidden, epochs_str)
    plt.savefig(filename)
    plt.close()
    print("Save figure {}".format(filename))


def reconstruct(images_data):
    images_recon = []
    for data in images_data:
        data = data.view(1, 784).to(device)
        if rbms[-1].n_visible == 784:
            feature = rbms[-1].sample_h(data)[1]
            recon = rbms[-1].sample_v(feature)[0]
        if rbms[-1].n_visible == 1000:
            feature = rbms[0].sample_h(data)[0]
            feature = rbms[-1].sample_h(feature)[1]
            feature = rbms[-1].sample_v(feature)[0]
            recon = rbms[0].sample_v(feature)[0]
        if rbms[-1].n_visible == 500:
            feature = rbms[0].sample_h(data)[0]
            feature = rbms[1].sample_h(feature)[0]
            feature = rbms[-1].sample_h(feature)[1]
            feature = rbms[-1].sample_v(feature)[0]
            feature = rbms[1].sample_v(feature)[0]
            recon = rbms[0].sample_v(feature)[0]
        if rbms[-1].n_visible == 250:
            feature = rbms[0].sample_h(data)[0]
            feature = rbms[1].sample_h(feature)[0]
            feature = rbms[2].sample_h(feature)[0]
            feature = rbms[-1].sample_h(feature)[1]
            feature = rbms[-1].sample_v(feature)[0]
            feature = rbms[2].sample_v(feature)[0]
            feature = rbms[1].sample_v(feature)[0]
            recon = rbms[0].sample_v(feature)[0]
        if rbms[-1].n_visible == 30:
            feature = rbms[0].sample_h(data)[0]
            feature = rbms[1].sample_h(feature)[0]
            feature = rbms[2].sample_h(feature)[0]
            feature = rbms[3].sample_h(feature)[0]
            feature = rbms[-1].sample_h(feature)[1]
            feature = rbms[-1].sample_v(feature)[0]
            feature = rbms[3].sample_v(feature)[0]
            feature = rbms[2].sample_v(feature)[0]
            feature = rbms[1].sample_v(feature)[0]
            recon = rbms[0].sample_v(feature)[0]
        img_recon = recon.cpu()
        img_recon = (img_recon*255).numpy().reshape(28, 28).astype('uint8')
        images_recon.append(img_recon)
    return images_recon


def trainRBM(epochs, epochs_viz):
    error_viz = []
    images_viz = [[(img*255).numpy().reshape(28, 28).astype('uint8') for img in images_data]]
    for epoch in range(1, epochs+1):
        epoch_error = 0.0
        since = time()
        for data, _ in train_loader:
            data = data.view(len(data), 784).to(device)
            if rbms[-1].n_visible == 784:
                batch_error = rbms[-1].constrastive_divergence(data).item()
            if rbms[-1].n_visible == 1000:
                feature = rbms[0].sample_h(data)[0]
                batch_error = rbms[-1].constrastive_divergence(feature).item()
            if rbms[-1].n_visible == 500:
                feature = rbms[0].sample_h(data)[0]
                feature = rbms[1].sample_h(feature)[0]
                batch_error = rbms[-1].constrastive_divergence(feature).item()
            if rbms[-1].n_visible == 250:
                feature = rbms[0].sample_h(data)[0]
                feature = rbms[1].sample_h(feature)[0]
                feature = rbms[2].sample_h(feature)[0]
                batch_error = rbms[-1].constrastive_divergence(feature).item()
            if rbms[-1].n_visible == 30:
                feature = rbms[0].sample_h(data)[0]
                feature = rbms[1].sample_h(feature)[0]
                feature = rbms[2].sample_h(feature)[0]
                feature = rbms[3].sample_h(feature)[0]
                batch_error = rbms[-1].constrastive_divergence(feature).item()
            epoch_error += batch_error
        spend = time() - since
        print("Epoch {}, error {:.4f} time {:.2f}s".format(epoch, epoch_error/len(train_loader), spend))
        error_viz.append(epoch_error/len(train_loader))
        if epoch in epochs_viz:
            images_viz.append(reconstruct(images_data))
        if epoch % 100 == 0:
            f = open("rbm_{}_{}_e{}.npz".format(rbms[-1].n_visible, rbms[-1].n_hidden, epoch), 'wb')
            np.savez(f, weight=rbms[-1].w, weight_momentum=rbms[-1].w_m,
                vbias=rbms[-1].vbias, vbias_momentum=rbms[-1].vbias_m,
                hbias=rbms[-1].hbias, hbias_momentum=rbms[-1].hbias_m)
            print("Save model parameters rbm_{}_{}_e{}.npz".format(rbms[-1].n_visible, rbms[-1].n_hidden, epoch))
    plt.plot(error_viz)
    plt.title("rbm_{}_{}_e{}_loss".format(rbms[-1].n_visible, rbms[-1].n_hidden, epochs))
    plt.savefig("rbm_{}_{}_e{}_loss.png".format(rbms[-1].n_visible, rbms[-1].n_hidden, epochs))
    plt.close()
    visualize(images_viz)


def viz_weights():
    ws = []
    epochs = [1, 10, 20, 50, 100]
    maps_inx = np.random.randint(0,100,size=100).tolist()
    for epoch in epochs:
        weights_file = "rbm_784_1000_e{}.npz".format(epoch)
        weights = np.load(weights_file)
        w = weights['weight']
        maps = []
        for inx in maps_inx:
            max = np.max(w[:,inx])
            map = ((max - w[:,inx])*255).reshape(28, 28).astype('uint8')
            maps.append(map)
        fig, axes = plt.subplots(10, 10)
        for i in range(10):
            for j in range(10):
                axes[i][j].imshow(maps[i*10+j], cmap=plt.cm.gray)
        for ax in axes.ravel():
            ax.set_axis_off()
        filename = "rbm_784_1000_e{}_map.png".format(epoch)
        plt.savefig(filename)
        plt.close()
        print("Save figure {}".format(filename))

def viz_map():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    epochs = [1, 10, 20, 50, 100, 200]
    for epoch in epochs:
        rbm = RBM(784, 1000, pretrained="rbm_784_1000_e{}.npz".format(epoch))
        viz = {}
        for j in range(10):
            im = images_data[j]
            maps = [(im*255).numpy().reshape(28, 28).astype('uint8')]
            data = im.view(-1, 784).to(device)
            feature = rbm.sample_h(data)[0].cpu().numpy()
            inx = np.argsort(feature)[0,-10:]
            coeff = feature[0, inx]
            print(coeff)
            for i in inx:
                w = rbm.w[:,i].cpu().numpy()
                max = np.max(w)
                map = ((max - w)*255).reshape(28, 28).astype('uint8')
                maps.append(map)
            viz[j] = maps
        print(viz[0][0].shape)
        fig, axes = plt.subplots(10, 11)
        for i in range(10):
            for j in range(11):
                axes[i][j].imshow(viz[i][j], cmap=plt.cm.gray)
        for ax in axes.ravel():
            ax.set_axis_off()
        filename = "rbm_784_1000_e{}_map10.png".format(epoch)
        plt.savefig(filename)
        plt.close()
        print("Save figure {}".format(filename))


def main():
    train_dataset = torchvision.datasets.MNIST(root='data/mnist', train=True,
        transform=torchvision.transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
    test_dataset = torchvision.datasets.MNIST(root='data/mnist', train=False,
        transform=torchvision.transforms.ToTensor(), download=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
    images_data = [test_dataset[i][0] for i in range(10)]
    rbm_names = ['rbm_784_1000', 'rbm_1000_500', 'rbm_500_250', 'rbm_250_30', 'rbm_30_2']
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    rbms = [RBM(784, 1000, pretrained="rbm_784_1000_e200.npz"),
            RBM(1000, 500, pretrained="rbm_1000_500_e300.npz"),
            RBM(500, 250, pretrained="rbm_500_250_e400.npz"),
            RBM(250, 30, pretrained="rbm_250_30_e500.npz")]
    for i in range(4, 5):
        visible, hidden = rbm_names[i].split('_')[1:]
        rbm = RBM(int(visible), int(hidden))
        rbm.lr *= 10
        rbms.append(rbm)
        start_time = time()
        epochs = 400
        epochs_viz = np.linspace(1, epochs, 10).astype('uint16').tolist()
        trainRBM(epochs, epochs_viz)
        end_time = time()
        print("Trainning for {} total time is {:.2f}s".format(rbm_names[i], end_time-start_time))
    test_features = np.zeros(((len(test_loader)-1)*128, 2))
    test_labels = np.zeros((len(test_loader)-1)*128)
    for i, (data, label) in enumerate(test_loader):
        if i == (len(test_loader) - 1):
            break
        data = data.view(len(data), 784).to(device)
        feature = rbms[0].sample_h(data)[0]
        feature = rbms[1].sample_h(feature)[0]
        feature = rbms[2].sample_h(feature)[0]
        feature = rbms[3].sample_h(feature)[0]
        feature = rbms[4].sample_h(feature)[0]
        test_features[i*128:(i+1)*128, :] = feature.cpu().numpy()
        test_labels[i*128:(i+1)*128] = label.numpy()
    plt.scatter(test_features[:, 0], test_features[:, 1], c=test_labels)
    plt.savefig("rbm_mnist_visualize.png")
    plt.close()


if __name__ == '__main__':
    main()
