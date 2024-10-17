import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
import matplotlib.pyplot as plt
from data_convert import DataLoader, Dataset
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
#from torcheval.metrics import R2Score
def Exact_plot(epoch, i, myz_true):
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

    fig = plt.gcf()
    fig.savefig('./Epoch_({})_Exact_{}FDuplot.png'.format(epoch + 1, i), bbox_inches='tight', pad_inches=0.02)
    # plt.show()

def Exact_Fplot(epoch, i, myz_true):
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

    fig = plt.gcf()
    fig.savefig('./Epoch_({})_Exact_{}FFDuplot.png'.format(epoch + 1, i), bbox_inches='tight', pad_inches=0.02)
    # plt.show()

def Pred_plot(epoch, i, z_out):
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
    fig.savefig('./Epoch_({})_Pred_{}FDuplot1000_onlyf.png'.format(epoch + 1,i), bbox_inches='tight', pad_inches=0.02)


def Error_plot(epoch, i, z_out, z_true):
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
    fig.savefig('./Epoch_({})_Error_{}FDuplot1000_onlyf.png'.format(epoch + 1,i), bbox_inches='tight', pad_inches=0.02)


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
    data_size = 1000
    
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
    


def test(dataloader, Encoder,model,Decoder, loss_fn, device, epoch,metric=None):
    # num_batches = len(dataloader)
    num_batches = 100//32
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for bcs, soln in dataloader:
            bcs, soln = bcs.to(device), soln.to(device)
            enc1, enc2, enc3, enc4, center,final=Encoder(bcs)
            #print("center",enc1.shape, enc2.shape, enc3.shape, enc4.shape, center.shape)
            center1 = model(center).reshape(center.shape)
            #print("center1",center1.shape)
            pred = Decoder(enc1, enc2, enc3, enc4,center1)
            #print(pred.shape)
            test_loss += loss_fn(pred, soln).item()
            if metric is not None:
                metric.update(pred.flatten(), soln.flatten())
        if ((epoch+1) % 10 == 0):
            for i in range(0,soln.shape[0],1000):
                # Exact_plot(epoch, i, soln[i][0])
                # Exact_Fplot(epoch, i, bcs[i][0])
                Pred_plot(epoch, i, pred[i][0])
                Error_plot(epoch, i, pred[i][0],soln[i][0])
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>6g}")
    return test_loss


def train_and_test(Encoder,model,Decoder, train_dataset, test_dataset,train_size,test_size,
                   params, summary_writer, device,batch_size,load_path,load_pathDs2,load_pathEs2,save_lossfile):
    """ Trains a model with given parameters and the given dataset. """

    # Mean squared error loss - the sum of squares is divided by the number of samples.
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        [{'params': model.parameters(),'lr':0.0001},
                {'params': Encoder.parameters(),'lr':0.001},
        {'params': Decoder.parameters(),'lr':0.0001}]
        )
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
            enc1, enc2, enc3, enc4, center,final=Encoder(bcs)
            #print("center",enc1.shape, enc2.shape, enc3.shape, enc4.shape, center.shape)
            center1 = model(center).reshape(center.shape)
            #print("center1",center1.shape)
            pred = Decoder(enc1, enc2, enc3, enc4,center1)
            #print(pred.shape)
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
        

        testerror = test(test_dataloader, Encoder,model,Decoder, loss_fn, device,epoch)
        MSE_Test.append(testerror)
        torch.save(model.state_dict(), load_path)
        torch.save(Decoder.state_dict(), load_pathDs2)
        torch.save(Encoder.state_dict(), load_pathEs2)
    print("写入loss!")
    with open(save_lossfile, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(MSE_Test)
        writer.writerow(MSE_Train)
        



def validate(Encoder,model,Decoder, validate_dataset, validate_size,batch_size, device):
    #metric = R2Score()
    validate_loader = DataLoader(Dataset(validate_dataset.F, validate_dataset.datas),validate_size,
                         batch_size)
    #validate_loader = TorchDataLoader(validate_dataset, batch_size=batch_size)
    loss_fn = torch.nn.MSELoss()
    testerror=test(validate_loader, Encoder,model,Decoder, loss_fn, device, 10001)
    return testerror
