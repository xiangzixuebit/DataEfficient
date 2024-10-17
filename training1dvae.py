import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
import matplotlib.pyplot as plt
from data_convert import DataLoader1d, Dataset
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
def pred_plot(epoch,i, xx, yy,zz):
    fig = plt.figure()
    font1 = {
             'weight': 'normal',
             'size': 18,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax = fig.add_subplot(111)
    ax.plot(xx, yy, 'b-', linewidth=2, label='Predicted')
    ax.plot(xx, zz, 'r--', linewidth=2, label='Exact')
    ax.set_xlabel('x', font2)
    ax.set_ylabel('u', font1)
    #plt.yscale('log')
    #plt.ylim(1e-1, 1e5)
    ax.legend(loc='best', prop=font1)

    fig = plt.gcf()
    fig.savefig('./Epoch{}u{}VAE.png'.format(epoch,i), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def pred_fplot(epoch,i, xx, yy,zz):
    fig = plt.figure()
    font1 = {
             'weight': 'normal',
             'size': 18,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax = fig.add_subplot(111)
    ax.plot(xx, yy, 'b-', linewidth=2, label='Predicted')
    ax.plot(xx, zz, 'g--', linewidth=2, label='Exact')
    ax.set_xlabel('x', font2)
    ax.set_ylabel('f', font1)
    #plt.yscale('log')
    #plt.ylim(1e-1, 1e5)
    ax.legend(loc='best', prop=font1)

    fig = plt.gcf()
    fig.savefig('./Epoch{}f{}VAE.png'.format(epoch,i), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def train(dataloader, model, loss_fn,
          optimizer, i_epoch, summary_writer, device):
    """ Trains a model for one epoch.

    @param dataloader  The Torch DataLoader object which provides access to batches of data.
    @param encoder_model  The model to be trained.
    @param decoder_model  The model to be trained.
    @param loss_fn  Object representing the type of loss function, eg., a torch.nn.MSELoss object.
    @param optimizer  The opimitizer, eg., a torch.optim.Adam object.
    @param i_epoch  The current epoch index.
    @param summary_writer  torch.utils.tensorboard SummaryWriter object.
    @param device  The hardware device on which to run the training.
    """
    # data_size = len(dataloader.dataset)
    data_size = 900
    
    for i_batch, (bcs, soln) in enumerate(dataloader):
        #print(bcs)
        bcs, soln = bcs.to(device), soln.to(device)
        if i_epoch == 0 and i_batch == 0:
            print(f" Shape of bcs [N, C, H, W]: {bcs.shape}")
            print(f" Shape of solution: {soln.shape} {soln.dtype}")
        pred = model(bcs)
        # print(pred.shape)
        loss = loss_fn(pred, soln)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_batch % 64 == 0:
            loss = loss.item()
            current = (i_batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{data_size:>5d}]")
            summary_writer.add_scalar('training loss', loss, i_epoch * len(dataloader) + i_batch)
    


def test(dataloader, encoder, decoder, loss_fn, device, epoch,metric=None):

    mse = torch.nn.MSELoss()
    encoder.eval()
    decoder.eval()
    test_loss = 0.0
    with torch.no_grad():
        for bcs, soln in dataloader:
            bcs, soln = bcs.to(device), soln.to(device)
            mu, log_sigma = encoder(bcs)
            sigma = torch.exp(log_sigma)
            # sampling by re-parameterization technique
            z = mu + sigma * torch.randn_like(mu)
            x_hat = decoder(z)
        if ((epoch+1) % 100 == 0):
            for i in range(0,soln.shape[0],1000):
                xx = np.linspace(0, 1, 101)
                pred_plot(epoch, i, xx, x_hat[i].cpu().detach().numpy(),bcs[i].cpu().detach().numpy())
                # pred_fplot(epoch, i, xx, z[i].cpu().detach().numpy(),bcs[i].cpu().detach().numpy())
            if metric is not None:
                metric.update(x_hat.flatten(), soln.flatten())


def get_loss(encoder, decoder, x, x_target):
    """

    :param encoder:
    :param decoder:
    :param x: input
    :param x_hat: target
    :param dim_img:
    :param dim_z:
    :param n_hidden:
    :param keep_prob:
    :return:
    """
    batchsz = x.size(0)
    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    # y = torch.clamp(y, 1e-8, 1 - 1e-8)

    marginal_likelihood = -torch.pow(x_target - y, 2).sum() / batchsz \
                          -torch.pow((x_target[:,1:]-x_target[:,:-1])-(y[:,1:]-y[:,:-1]), 2).sum()/batchsz
    # print(marginal_likelihood2.item(), marginal_likelihood.item())

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence
    loss = -ELBO
    return y, z, loss, marginal_likelihood, KL_divergence

def train_and_test(encoder, decoder, train_dataset, test_dataset,train_size,test_size,
                   params, device,batch_size,load_pathencoder,load_pathdecoder,save_lossfile):
    """ Trains a model with given parameters and the given dataset. """

    # Mean squared error loss - the sum of squares is divided by the number of samples.
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([
                {'params': encoder.parameters(),'lr':0.0001},
        {'params': decoder.parameters(),'lr':0.0001}]
        )
    train_dataloader = DataLoader1d(Dataset(train_dataset.F, train_dataset.datas), train_size,
                         batch_size)
    test_dataloader = DataLoader1d(Dataset(test_dataset.F, test_dataset.datas),test_size,
                         batch_size)
    # train_dataloader = TorchDataLoader(train_dataset, batch_size=params["batch_size"],
    #                                    shuffle=params["shuffle"])
    # test_dataloader = TorchDataLoader(test_dataset, batch_size=params["batch_size"])
    MSE_Train = []
    MSE_Test = []

    for epoch in range(params["num_epochs"]):
        for i_batch, (bcs, soln) in enumerate(train_dataloader):
            #print(bcs)
            bcs, soln = bcs.to(device), soln.to(device)
            if epoch == 0 and i_batch == 0:
                print(f" Shape of bcs [N, C, H, W]: {bcs.shape}")
                print(f" Shape of solution: {soln.shape} {soln.dtype}")
            y, z, tot_loss, loss_likelihood, loss_divergence = \
                                    get_loss(encoder, decoder, bcs, soln)
            # [mu_z, logvar_z, z] = encoder(bcs)
            # x_hat = decoder(z)

            # # compute loss & backpropagation 
            # reconstruction_loss = mse(bcs, x_hat)
            # kl_loss = -1 - logvar_z + torch.square(mu_z) + torch.exp(logvar_z)
            # kl_loss = 0.5 * torch.sum(kl_loss)
            # # label_loss = torch.divide(0.5 * torch.square(mu_y - soln), torch.exp(logvar_y)) + 0.5 *logvar_y
            # vae_loss = torch.mean(reconstruction_loss + kl_loss)
            
            MSE_Train.append(tot_loss.item())
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

            if epoch % 100==0 and i_batch % batch_size == 0:
               print("L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                                            tot_loss.item(), loss_likelihood.item(), loss_divergence.item()))
        test(test_dataloader, encoder, decoder, loss_fn, device,epoch)
        # MSE_Test.append(testerror)
    torch.save(encoder.state_dict(), load_pathencoder)
    torch.save(decoder.state_dict(), load_pathdecoder)
    print("写入loss!")
    with open(save_lossfile, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(MSE_Test)
        writer.writerow(MSE_Train)

def validate(modelvalidate_dataset, validate_size,batch_size, device):
    #metric = R2Score()
    validate_loader = DataLoader1d(Dataset(validate_dataset.F, validate_dataset.datas),validate_size,
                         batch_size)
    #validate_loader = TorchDataLoader(validate_dataset, batch_size=batch_size)
    loss_fn = torch.nn.MSELoss()
    testerror=test(validate_loader, model, loss_fn, device, 10001)
    return testerror
