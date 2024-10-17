import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
import matplotlib.pyplot as plt
from data_convert import DataLoader, Dataset
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
#from torcheval.metrics import R2Score
def Exact_plot(epoch, i,j, myz_true):
    fig, ax = plt.subplots(figsize=(9,9))
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    h0 = ax.imshow(myz_true.cpu().detach().numpy(), interpolation='nearest', cmap='viridis',
                      extent=[-0.02, 2.00, -0.02, 2.00],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$y$', font2)
    ax.set_ylabel('$x$', font2)
    # ax[0].set_aspect('equal', 'box')
    ax.set_title('Exact', fontsize=17)

    fig = plt.gcf()
    fig.savefig('./Epoch_({})_Exact_{}_{}plotu.png'.format(epoch + 1, i,j), bbox_inches='tight', pad_inches=0.02)
    # plt.show()

def Exact_Fplot(epoch, i,j, myz_true):
    fig, ax = plt.subplots(figsize=(9,9))
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    h0 = ax.imshow(myz_true.cpu().detach().numpy(), interpolation='nearest', cmap='viridis',
                      extent=[-0.02, 2.00, -0.02, 2.00],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$y$', font2)
    ax.set_ylabel('$x$', font2)
    # ax[0].set_aspect('equal', 'box')
    ax.set_title('F', fontsize=17)

    fig = plt.gcf()
    fig.savefig('./Epoch_({})_Exact_{}_{}Fplotu.png'.format(epoch + 1, i,j), bbox_inches='tight', pad_inches=0.02)
    # plt.show()

def Pred_plot(epoch, i,j, z_out):
    fig, ax = plt.subplots(figsize=(9,9))
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率

    h0 = ax.imshow(z_out.cpu().detach().numpy(), interpolation='nearest', cmap='viridis',
                      extent=[-0.02, 2.00, -0.02, 2.00],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$y$', font2)
    ax.set_ylabel('$x$', font2)
    # ax[0].set_aspect('equal', 'box')
    ax.set_title('Predicted', fontsize=17)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_Pred_{}plot.pdf'.format(epoch + 1,i), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./Epoch_({})_Pred_{}_{}plotu.png'.format(epoch + 1,i,j), bbox_inches='tight', pad_inches=0.02)


def Error_plot(epoch, i, j,z_out, z_true):
    fig, ax = plt.subplots(figsize=(9,9))
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率

    h0 = ax.imshow(np.abs(z_out.cpu().detach().numpy()-z_true.cpu().detach().numpy()), interpolation='nearest', cmap='viridis',
                      extent=[-0.02, 2.00, -0.02, 2.00],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$y$', font2)
    ax.set_ylabel('$x$', font2)
    # ax[0].set_aspect('equal', 'box')
    ax.set_title('Error', fontsize=17)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_Pred_{}plot.pdf'.format(epoch + 1,i), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./Epoch_({})_Error_{}_{}plotu.png'.format(epoch + 1,i,j), bbox_inches='tight', pad_inches=0.02)


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

        if i_batch % 32 == 0:
            loss = loss.item()
            current = (i_batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{data_size:>5d}]")
            summary_writer.add_scalar('training loss', loss, i_epoch * len(dataloader) + i_batch)
    


def test(dataloader, model, loss_fn, device, epoch,metric=None):
    # num_batches = len(dataloader)
    num_batches = 100//32
    model.eval()
    test_loss = 0.0
    j=0
    with torch.no_grad():
        for bcs, soln in dataloader:
            bcs, soln = bcs.to(device), soln.to(device)
            pred = model(bcs)
            test_loss += loss_fn(pred, soln).item()

            if ((epoch+1) % 1 == 0):
                for i in range(0,soln.shape[0],100):
                    print(i)
                    Exact_plot(epoch, i, j,soln[i][0])
                    Exact_Fplot(epoch, i, j,bcs[i][0])
                    Pred_plot(epoch, i,j, pred[i][0])
                    Error_plot(epoch, i,j, pred[i][0],soln[i][0])
                    print(j)
                    j=j+1

            if metric is not None:
                metric.update(pred.flatten(), soln.flatten())
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>6g}")
    return test_loss


def train_and_test(model, train_dataset, test_dataset,train_size,test_size,
                   params, summary_writer, device,batch_size,load_path,save_lossfile):
    """ Trains a model with given parameters and the given dataset. """

    # Mean squared error loss - the sum of squares is divided by the number of samples.
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"])
    train_dataloader = DataLoader(Dataset(train_dataset.F, train_dataset.datas), train_size,
                         batch_size)
    test_dataloader = DataLoader(Dataset(test_dataset.F, test_dataset.datas),test_size,
                         batch_size)
    # train_dataloader = TorchDataLoader(train_dataset, batch_size=params["batch_size"],
    #                                    shuffle=params["shuffle"])
    # test_dataloader = TorchDataLoader(test_dataset, batch_size=params["batch_size"])
    MSE_Train = []
    MSE_Test = []
    for epoch in range(params["num_epochs"]):
        print(f"\nEpoch {epoch+1}")
        for i_batch, (bcs, soln) in enumerate(train_dataloader):
            #print(bcs)
            bcs, soln = bcs.to(device), soln.to(device)
            if epoch == 0 and i_batch == 0:
                print(f" Shape of bcs [N, C, H, W]: {bcs.shape}")
                print(f" Shape of solution: {soln.shape} {soln.dtype}")
            pred = model(bcs)
            # print(pred.shape)
            loss = loss_fn(pred, soln)
            MSE_Train.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch % 32 == 0:
                loss = loss.item()
                current = (i_batch + 1)
                print(f"loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")
                summary_writer.add_scalar('training loss', loss, epoch * len(train_dataloader) + i_batch)
        

        testerror = test(test_dataloader, model, loss_fn, device,epoch)
        MSE_Test.append(testerror)
    torch.save(model.state_dict(), load_path)
    print("写入loss!")
    with open(save_lossfile, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(MSE_Test)
        writer.writerow(MSE_Train)
        



def validate(model, validate_dataset, validate_size,batch_size, device):
    #metric = R2Score()
    validate_loader = DataLoader(Dataset(validate_dataset.F, validate_dataset.datas),validate_size,
                         batch_size)
    #validate_loader = TorchDataLoader(validate_dataset, batch_size=batch_size)
    loss_fn = torch.nn.MSELoss()
    testerror=test(validate_loader, model, loss_fn, device, 10001)
    return testerror
