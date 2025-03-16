
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR
import warnings
import argparse
# from .models.extract_feature import SurfNN
from datasets.surface_dataset_ANDI_tau_vtk import My_dHCP_Data_tau
import yaml
from utils import *
import importlib
import geomloss
import  shutil
from loss.euclidean_distance_wd_loss import euclidean_distance_wd_loss

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
parser.add_argument('--dist_url', type=str, default='env://')
args = parser.parse_args()


with open(args.config, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
    for key, value in config.items():
        parser.add_argument('--' + key, type=type(value), default=value)

    args = parser.parse_args()


    print("Arguments:", args)

warnings.filterwarnings("ignore")

def cost_func(a, b, p=2, metric='cosine'):
    """ a, b in shape: (B, N, D) or (N, D)
    """
    assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
    if metric=='euclidean' and p==1:
        return geomloss.utils.distances(a, b)
    elif metric=='euclidean' and p==2:
        return geomloss.utils.squared_distances(a, b)
    else:
        if a.dim() == 3:
            x_norm = a / a.norm(dim=2)[:, :, None]
            y_norm = b / b.norm(dim=2)[:, :, None]
            M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
        elif a.dim() == 2:
            x_norm = a / a.norm(dim=1)[:, None]
            y_norm = b / b.norm(dim=1)[:, None]
            M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
        M = pow(M, p)
        return M

entreg = 0.1
metric = 'euclidean'
OTLoss = euclidean_distance_wd_loss(
    loss='sinkhorn', p=1,
    cost=lambda a, b: cost_func(a, b, p=1, metric=metric),
    blur=entreg**(1/1), backend='tensorized',size= 4000)
# pW = OTLoss(a, b)


if __name__ == '__main__':


    log_columns = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'val_loss', 'val_acc']
    input_dir = args.input_dir
    val_dir = args.val_dir
    test_dir = args.test_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    in_features = args.in_features
    feature_size = args.feature_size
    milestones = args.milestones
    resume = args.resume
    use_cuda = args.use_cuda
    use_DDP = args.use_DDP
    size = args.submatrix_length
    resume_log = args.resume_log
    total_epochs = args.total_epochs
    save_test_result = args.save_test_result
    save_result = args.save_result
    gamma = args.gamma
    use_sche = args.use_sche
    if use_DDP:
        init_distributed_mode(args)

    #---------------------------------------------------------
    lr_pe = args.lr_pe  # 0.0002
    lr_pro = args.lr_pro  # 0.0001
    lr_g = args.lr_g  # 0.003
    lr_d = args.lr_d  # 0.003
    #---------------------------------------------------------

    make_dir(args.output_dir)
    output_dir = check_output_dir(args.output_dir)
    test_result_path = join_a_and_b_dir(output_dir, args.test_result_name)
    models_save_path = join_a_and_b_dir(output_dir, 'models_save')
    make_dir(models_save_path)
    make_dir(test_result_path)
    shutil.copy(args.config, os.path.join(test_result_path, args.config.split("/")[-1]))
    if not use_DDP:
        args.device = torch.device('cuda:' + str(args.cuda_device) if args.use_cuda else 'cpu')
    device = args.device
    #-------------------------------------------------------------------------------
    train_dataset = My_dHCP_Data_tau(input_dir)
    val_dataset = My_dHCP_Data_tau(val_dir)
    test_dataset = My_dHCP_Data_tau(test_dir)
    if use_DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                                   sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                                 sampler=val_sampler)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                                  sampler=test_sampler)

    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # -------------------------------------------------------------------------------
    model_G_params = args.model_G_params
    model_module_G = importlib.import_module(f'models.{config["model_G_dir"]}')
    model_G = getattr(model_module_G, args.model_G)
    model_G = model_G(**model_G_params)

    model_D_params = args.model_D_params
    model_module_D = importlib.import_module(f'models.{config["model_D_dir"]}')
    model_D = getattr(model_module_D, args.model_D)
    model_D = model_D(**model_D_params)
    # -------------------------------------------------------------------------------
    # model_propcess_params = args.model_propcess_params
    model_propcess_module = importlib.import_module(f'models.{config["model_propcess_dir"]}')
    SurfNN = getattr(model_propcess_module, args.model_propcess)
    model_propcess = SurfNN()

    model_PE_params = args.model_PE_params
    model_PE_module = importlib.import_module(f'models.{config["model_PE_dir"]}')
    DNN = getattr(model_PE_module, args.model_PE)
    model_PE = DNN(**model_PE_params)

    optimizer_PE = Adam(model_PE.parameters(), lr=lr_pe)
    optimizer_PROPCESS = Adam(model_propcess.parameters(), lr=lr_pro)
    optimizer_G = Adam(model_G.parameters(), lr=lr_g, betas=(0.5, 0.999))  
    optimizer_D = Adam(model_D.parameters(), lr=lr_d, betas=(0.5, 0.999))  
    if use_sche:
        scheduler_g = MultiStepLR(optimizer_G, milestones=milestones, gamma=gamma)
        scheduler_d = MultiStepLR(optimizer_D, milestones=milestones, gamma=gamma)

    mse_loss = nn.MSELoss()
    # mae_loss = nn.MAELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    print(test_result_path)
    start_epoch = 1

    if resume != "None":
        print('resume: ' + resume)
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        optimizer_G.load_state_dict(checkpoint['opt_g'])
        optimizer_D.load_state_dict(checkpoint['opt_d'])
        optimizer_PE.load_state_dict(checkpoint['opt_pe'])
        optimizer_PROPCESS.load_state_dict(checkpoint['opt_pro'])

        model_D.load_state_dict(checkpoint['model_D'])
        model_G.load_state_dict(checkpoint['model_G'])
        model_PE.load_state_dict(checkpoint['model_PE'])
        model_propcess.load_state_dict(checkpoint['model_propcess'])

    if use_cuda:
        model_D = model_D.to(args.device)  # device
        model_G = model_G.to(args.device)
        model_propcess = model_propcess.to(args.device)
        model_PE = model_PE.to(args.device)
        if use_DDP:
            model_D = nn.parallel.DistributedDataParallel(model_D, device_ids=[args.device],
                                                          find_unused_parameters=True)  # device_ids=[args.device]
            model_G = nn.parallel.DistributedDataParallel(model_G, device_ids=[args.device],
                                                          find_unused_parameters=True)
            model_propcess = nn.parallel.DistributedDataParallel(model_propcess, device_ids=[args.device],
                                                                 find_unused_parameters=True)
            model_PE = nn.parallel.DistributedDataParallel(model_PE, device_ids=[args.device],
                                                           find_unused_parameters=True)

    else:
        model_D = model_D.cpu()
        model_G = model_G.cpu()
        model_propcess = model_propcess.cpu()
        model_PE = model_PE.cpu()


    # Training
    def train(data_loader):

        losses = AverageMeter()
        g_losses = AverageMeter()
        d_losses = AverageMeter()
        MAES = AverageMeter()
        mae_list = []
        model_propcess.train()
        model_D.train()
        model_G.train()
        model_PE.train()

        bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, samples in bar:
            if len(samples) == 0:
                print('error', batch_idx)
                break

            # data install
            inputs_start = samples['mgh0'].float()  # fake
            inputs_next = samples['mgh1'].float()  # real
            v0 = samples['v0'].float()  # real
            v1 = samples['v1'].float()  # real
            adj = samples['f0']
            adj1 = samples['f1']

            if use_cuda:
                inputs_start = inputs_start.to(device)
                inputs_next = inputs_next.to(device)
                v0 = v0.to(device)
                v1 = v1.to(device)
                adj = adj.to(device)
                adj1 = adj1.to(device)

            inputs_start = inputs_start.squeeze(0)
            inputs_next = inputs_next.squeeze(0)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real data (we regard the next state as the real data)
            model_D.zero_grad()
            inputs_next1, adj_matrix1, distance1, distance1_ = model_propcess(inputs_next, adj1, v1)
            U0_r = model_PE(inputs_next1)  # noise data
            U0_r = U0_r.squeeze(0)
            in_features1 = U0_r.size(0)
            U0_r_expanded = U0_r.expand(in_features1, -1)
            zeros_m = torch.zeros((1, U0_r_expanded.size(1))).to(device)
            U0_r_expanded = torch.cat((U0_r_expanded, zeros_m), dim=0)

            tmp_list = []
            chunk_size = size
            for i in range(0, in_features1, chunk_size):
                tmp_index = adj_matrix1[i:i + chunk_size, :]
                chunk_P_r = U0_r_expanded[tmp_index[:, 0].long(), :] - U0_r_expanded[tmp_index[:, :].long().t(),:].squeeze(2).t()
                tmp = chunk_P_r.sum(dim=1)
                tmp_list.append(tmp)
            tmp_tensor = torch.concat(tmp_list, dim=0).unsqueeze(1)
            del tmp_list, tmp
            U11_r = U0_r + tmp_tensor
            output = model_D(U11_r, adj_matrix1, distance1)
            errD_real = OTLoss(distance1_, adj_matrix1, output, inputs_next)
            print("finish errD_real", errD_real)
            errD_real.backward()
            # ------------------------------------------------------------------------
            inputs_start0, adj_matrix0, distance0, distance0_ = model_propcess(inputs_start, adj, v0)
            U00 = model_PE(inputs_start0)  # noise data
            fake = model_G(U00.squeeze(0), adj_matrix0, distance0)

            U0_fake, adj_matrix_fake, distance_fake, distance_fake_ = model_propcess(fake.squeeze(0), adj1, v1)
            U0 = model_PE(U0_fake)
            U0 = U0.clone()
            # on fake
            U0_expanded = U0.squeeze(0).expand(in_features1, -1)
            zeros_m = torch.zeros((1, U0_expanded.size(1))).to(device)

            U0_expanded = torch.cat((U0_expanded, zeros_m), dim=0)

            tmp_list = []
            chunk_size = size
            for i in range(0, in_features1, chunk_size):
                tmp_index = adj_matrix_fake[i:i + chunk_size, :]
                chunk_P_r = U0_expanded[tmp_index[:, 0].long(), :] - U0_expanded[tmp_index[:, :].long().t(), :].squeeze(2).t()
                tmp = chunk_P_r.sum(dim=1)
                # print("tmp", tmp.size())
                tmp_list.append(tmp)
            tmp_tensor = torch.concat(tmp_list, dim=0).unsqueeze(1)
            del tmp_list, tmp
            U11 = U0 + tmp_tensor

            U1_fake = model_D(U11.squeeze(0), adj_matrix1, distance_fake)
            errD_fake = OTLoss(distance_fake_, adj_matrix1,U1_fake, fake)
            errD_fake.backward()
            D_loss = errD_real

            # Update D
            optimizer_D.step()
            optimizer_PROPCESS.step()
            optimizer_PE.step()

            # ------------------------------------------------------------------------
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            model_G.zero_grad()
            inputs_start00, adj_matrix, distance0, distance0_ = model_propcess(inputs_start, adj, v0)
            # adj_matrix1 = adj_matrix1.to(device)
            # on real
            U0 = model_PE(inputs_start00)
            U0_expand = U0.squeeze(0)
            in_features1 = U0_expand.size(0)
            zeros_m = torch.zeros((1, U0_expand.size(1))).to(device)

            U0_expand = torch.cat((U0_expand, zeros_m), dim=0)
            # print("U0_expand",U0_expand.shape)

            tmp_list = []
            chunk_size = size
            for i in range(0, in_features1, chunk_size):
                tmp_index = adj_matrix[i:i + chunk_size, :]
                chunk_P_r = U0_expand[tmp_index[:, 0].long(), :] - U0_expand[tmp_index[:, :].long().t(), :].squeeze(
                    2).t()
                tmp = chunk_P_r.sum(dim=1)
                # print("tmp", tmp.size())
                tmp_list.append(tmp)
            tmp_tensor = torch.concat(tmp_list, dim=0).unsqueeze(1)
            del tmp_list, tmp
            # print(tmp_tensor.shape)
            # print("tmp_tensor", tmp_tensor.shape)
            U11 = U0 + tmp_tensor
            fake = model_G(U11.squeeze(0), adj_matrix, distance0)  # U11.squeeze(0)
            G_fake_loss = OTLoss(distance0_, adj_matrix,fake, inputs_next)
            # G_loss = huber_loss(output_fake, inputs_next.squeeze(0)) * 2
            G_fake_loss.backward()
            print("G_fake_loss", G_fake_loss)
            #---------------------------------------------

            inputs_start0, adj_matrix0, distance0, distance0_ = model_propcess(inputs_start, adj, v0)
            U0 = model_PE(inputs_start0)
            output_fake = model_G(U0.squeeze(0), adj_matrix0, distance0)
            fake_1, adj_fake1, distance_fake, distance_fake_ = model_propcess(output_fake.squeeze(0), adj1, v1)
            U0_g = model_PE(fake_1)

            U0_g_expanded = U0_g.squeeze(0).expand(in_features1, -1)
            zeros_m = torch.zeros((1, U0_g_expanded.size(1))).to(device)
            U0_g_expanded = torch.cat((U0_g_expanded, zeros_m), dim=0)
            tmp_list = []
            chunk_size = size
            for i in range(0, in_features1, chunk_size):
                tmp_index = adj_fake1[i:i + chunk_size, :]
                chunk_P_r = U0_g_expanded[tmp_index[:, 0].long(), :] - U0_g_expanded[tmp_index[:, :].long().t(),:].squeeze(2).t()
                tmp = chunk_P_r.sum(dim=1)
                tmp_list.append(tmp)
            tmp_tensor = torch.concat(tmp_list, dim=0).unsqueeze(1)
            del tmp_list, tmp
            U11_g = U0_g + tmp_tensor
            D_fake = model_D(U11_g.squeeze(0), adj_fake1, distance_fake)
            print("**end three calculate model_D")

            G_loss = OTLoss(distance_fake_, adj_fake1, output_fake, D_fake)
            G_loss.backward()
            optimizer_G.step()
            optimizer_PROPCESS.step()
            optimizer_PE.step()

            MAE = mean_absolute_error(torch.tensor(output_fake).cpu().numpy(), inputs_next.squeeze(0).cpu().numpy())
            loss = D_loss + G_loss

            losses.update(loss.item(), samples['v0'].size(0))
            g_losses.update(G_loss.item(), samples['v0'].size(0))
            d_losses.update(D_loss.item(), samples['v0'].size(0))
            # mae_losses.update(MAE_loss.item(), inputs.size(0))
            MAES.update(MAE.item(), 1)
            mae_list.append(MAE)

            if G_loss is None:
                bar.set_description(
                    f'loss: {losses.avg:.4f} ({losses.val:.4f}) | d_loss: {d_losses.avg:.4f} ({d_losses.val:.4f}) | mae: {MAES.avg:.4f} ({MAES.val:.4f})')

            else:
                bar.set_description(
                    f'loss: {losses.avg:.4f} | d_loss: {d_losses.avg:.4f} | g_loss: {g_losses.avg:.4f} | mae: {MAES.avg:.4f}')

            # break # one tests

        return losses.avg, MAES.avg, mae_list, g_losses.avg, d_losses.avg


    @torch.no_grad()
    def test(data_loader):
        losses = AverageMeter()
        MAES = AverageMeter()
        mae_list = []
        model_G.eval()
        model_propcess.eval()
        model_PE.eval()

        bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, samples in bar:

            inputs_start = samples['mgh0'].float()  # fake
            inputs_next = samples['mgh1'].float()  # real
            v0 = samples['v0'].float()  # real
            adj = samples['f0']

            if use_cuda:
                inputs_start = inputs_start.to(device)
                inputs_next = inputs_next.to(device)
                v0 = v0.to(device)
                adj = adj.to(device)

            inputs_start = inputs_start.squeeze(0)
            inputs_next = inputs_next.squeeze(0)
            inputs_start, adj_matrix, distance0, distance0_ = model_propcess(inputs_start, adj, v0)
            # on real
            U0 = model_PE(inputs_start)
            U0_expand = U0.squeeze(0)
            in_features1 = U0_expand.size(0)
            zeros_m = torch.zeros((1, U0_expand.size(1))).to(device)

            U0_expand = torch.cat((U0_expand, zeros_m), dim=0)

            tmp_list = []
            chunk_size = size
            for i in range(0, in_features1, chunk_size):
                tmp_index = adj_matrix[i:i + chunk_size, :]
                chunk_P_r = U0_expand[tmp_index[:, 0].long(), :] - U0_expand[tmp_index[:, :].long().t(), :].squeeze(
                    2).t()
                tmp = chunk_P_r.sum(dim=1)
                tmp_list.append(tmp)
            tmp_tensor = torch.concat(tmp_list, dim=0).unsqueeze(1)
            del tmp_list, tmp
            U11 = U0 + tmp_tensor
            fake = model_G(U11.squeeze(0), adj_matrix, distance0)  # U11.squeeze(0)

            MAE = mean_absolute_error(torch.tensor(fake).squeeze(0).cpu().numpy(),
                                      inputs_next.squeeze(0).cpu().numpy())
            MAES.update(MAE.item(), 1)
            mae_list.append(MAE)

            if MAE is None:
                bar.set_description(
                    # f'loss: {losses.avg:.4f} ({losses.val:.4f}) | acc: {accs.avg:.4f} ({accs.val:.4f})')
                    f'loss: {losses.avg:.4f} ({losses.val:.4f})  | mae: {MAES.avg:.4f} ({MAES.val:.4f})')

            else:
                bar.set_description(
                    f' mae: {MAES.avg:.4f}')
            # # break

        return MAES.avg, mae_list


    train_losses = []
    train_maes = []
    test_maes = []
    val_maes = []
    epochs = []
    best_mae = 100
    best_mae_test =100
    g_losses = []
    d_losses = []
    epochs = []

    for epoch in range(start_epoch, start_epoch + total_epochs):
        if use_sche:
            scheduler_d.step()
            scheduler_g.step()
        print('\nEpoch: %d' % epoch)
        print('---------------------------', optimizer_G.param_groups[0]['lr'])

        train_loss, train_mae, train_maess, g_loss, d_loss = train(train_loader)
        test_mae, test_maess = test(test_loader)
        val_mae, _ = test(val_loader)

        train_losses.append(train_loss)
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        train_maes.append(train_mae)

        # val_losses.append(val_loss)
        val_maes.append(val_mae)
        cur_lr_g = optimizer_G.param_groups[0]['lr']
        cur_lr_d = optimizer_D.param_groups[0]['lr']
        lr_ges = [float(cur_lr_g)] * len(train_maes)
        lr_des = [float(cur_lr_d)] * len(train_maes)
        epoches = [epoch] * len(train_maes)

        epochs.append(epoch)
        test_maes.append(test_mae)

        if test_mae < best_mae_test:
            print('best_test')
            best_mae_test = test_mae
            torch.save({'epoch': epoch,
                        'model_G': model_G.state_dict(),
                        'model_D': model_D.state_dict(),
                        'model_PE': model_PE.state_dict(),
                        "model_propcess": model_propcess.state_dict(),
                        'opt_g': optimizer_G.state_dict(),
                        'opt_D': optimizer_D.state_dict(),
                        'opt_pro': optimizer_PROPCESS.state_dict(),
                        'opt_pe': optimizer_PE.state_dict(),
                        # 'sche_g': scheduler_g.state_dict(),
                        # 'sche_D': scheduler_d.state_dict(),
                        }, os.path.join(models_save_path, "best.pt"))
            if save_test_result:

                epoches = [epoch] * len(train_maes)
                combine_matrix = np.array([epoches, train_maes, test_maes]).transpose()
                np.savetxt(os.path.join(test_result_path, f'test_mae.csv'),
                           combine_matrix, fmt='%f', delimiter=',')


        if save_result:
            plot_epochs(os.path.join(test_result_path, f'loss.svg'),
                        [train_losses, g_losses, d_losses],
                        epochs, xlabel='epoch', ylabel='loss', legends=['train_losses', 'g_losses', 'd_losses'],
                        max=False)
            plot_epochs(os.path.join(test_result_path, f'MAE.svg'),
                        [train_maes, test_maes, val_maes],
                        epochs, xlabel='epoch', ylabel='maes', legends=['train_maes', 'test_maes', 'val_maes'])

            pd.DataFrame([[epoch, train_mae, test_mae, val_mae]]).to_csv(
                os.path.join(test_result_path, f'log.csv'), mode='a', index=False,
                header=False)
            if use_DDP:
                if dist.get_rank() == 0:
                    torch.save({'epoch': epoch,
                                'model_G': model_G.state_dict(),
                                'model_D': model_D.state_dict(),
                                'model_PE': model_PE.state_dict(),
                                "model_propcess": model_propcess.state_dict(),
                                'opt_g': optimizer_G.state_dict(),
                                'opt_D': optimizer_D.state_dict(),
                                'opt_pro': optimizer_PROPCESS.state_dict(),
                                'opt_pe': optimizer_PE.state_dict(),
                                # 'sche_g': scheduler_g.state_dict(),
                                # 'sche_D': scheduler_d.state_dict(),
                                }, os.path.join(models_save_path, "%d_%.4f.pt" % (epoch, test_mae)))
            else:
                torch.save({'epoch': epoch,
                            'model_G': model_G.state_dict(),
                            'model_D': model_D.state_dict(),
                            'model_PE': model_PE.state_dict(),
                            "model_propcess": model_propcess.state_dict(),
                            'opt_g': optimizer_G.state_dict(),
                            'opt_D': optimizer_D.state_dict(),
                            'opt_pro': optimizer_PROPCESS.state_dict(),
                            'opt_pe': optimizer_PE.state_dict(),
                            # 'sche_g': scheduler_g.state_dict(),
                            # 'sche_D': scheduler_d.state_dict(),
                            }, os.path.join(models_save_path, "%d_%.4f.pt" % (epoch, test_mae)))

