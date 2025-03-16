from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import matplotlib as mpl
mpl.use('Agg')
import warnings
import argparse
from datasets.surface_dataset_ANDI_tau import My_dHCP_Data_tau
import yaml
from .utils import *
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Tau')
parser.add_argument('--noise_level', type=int, default=0)
args = parser.parse_args()

warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')


args = parser.parse_args()


with open(args.config, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
    for key, value in config.items():
        parser.add_argument('--' + key, type=type(value), default=value)


    args = parser.parse_args()

  
    print("Arguments:", args)

warnings.filterwarnings("ignore")


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
    disease_list = args.disease_list
    data_path = args.data_path
    save_path = args.save_path

    # ---------------------------------------------------------
    lr_pe = args.lr_pe  # 0.0002
    lr_pro = args.lr_pro  # 0.0001
    lr_g = args.lr_g  # 0.003
    lr_d = args.lr_d  # 0.003
    # ---------------------------------------------------------

    make_dir(args.output_dir)
    output_dir = check_output_dir(args.output_dir) # output_dir = "output_tau_mse_loss_DeepGCN_0.001_0.006_0.0002_0.0001_0" #change
    test_result_path = join_a_and_b_dir(output_dir, args.test_result_name)
    models_save_path = join_a_and_b_dir(output_dir, 'models_save')
    make_dir(models_save_path)
    make_dir(test_result_path)
    if not use_DDP:
        device = torch.device('cuda:' + str(args.cuda_device) if args.use_cuda else 'cpu')

    mae_all = 0
    num = 0

    # -------------------------------------------------------------------------------

    model_G_params = args.model_G_params
    model_module_G = importlib.import_module(f'model.{config["model_G_dir"]}')
    model_G = getattr(model_module_G, args.model_G)
    model_G = model_G(**model_G_params)

    model_D_params = args.model_D_params
    model_module_D = importlib.import_module(f'model.{config["model_D_dir"]}')
    model_D = getattr(model_module_D, args.model_D)
    model_D = model_D(**model_D_params)
    # -------------------------------------------------------------------------------
    # model_propcess_params = args.model_propcess_params
    model_propcess_module = importlib.import_module(f'model.{config["model_propcess_dir"]}')
    SurfNN = getattr(model_propcess_module, args.model_propcess)
    model_propcess = SurfNN()

    model_PE_params = args.model_PE_params
    model_PE_module = importlib.import_module(f'model.{config["model_PE_dir"]}')
    DNN = getattr(model_PE_module, args.model_PE)
    model_PE = DNN(**model_PE_params)

    if resume:
        checkpoint = torch.load(resume, map_location=device)
        # print('resume: ', resume)
        if use_DDP:
            new_model_d_sd = rebuild_model_state_dict(checkpoint['model_D'])
            new_model_g_sd = rebuild_model_state_dict(checkpoint['model_G'])
            new_model_pe_sd = rebuild_model_state_dict(checkpoint['model_PE'])
            new_model_pro_sd = rebuild_model_state_dict(checkpoint['model_propcess'])  # need to use
            model_D.load_state_dict(new_model_d_sd)
            model_G.load_state_dict(new_model_g_sd)
            model_PE.load_state_dict(new_model_pe_sd)
            model_propcess.load_state_dict(new_model_pro_sd)
        else:
            model_D.load_state_dict(checkpoint['model_D'])
            model_G.load_state_dict(checkpoint['model_G'])
            model_PE.load_state_dict(checkpoint['model_PE'])
            model_propcess.load_state_dict(checkpoint['model_propcess'])
            start_epoch = checkpoint['epoch']

    if use_cuda:
        model_D = model_D.to(device)  # device
        model_G = model_G.to(device)
        model_propcess = model_propcess.to(device)
        model_PE = model_PE.to(device)
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


    @torch.no_grad()
    def test(data_loader):
        losses = AverageMeter()
        MAES = AverageMeter()
        mae_list = []
        model_G.eval()
        model_propcess.eval()
        model_PE.eval()

        bar = tqdm(enumerate(data_loader), total=len(data_loader))
        # for batch_idx, (inputs, targets, structual) in bar:
        for batch_idx, samples in bar:

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
                adj = adj.to(device)

            inputs_start = inputs_start.squeeze(0)
            inputs_next = inputs_next.squeeze(0)

            inputs_start, adj_matrix, distance0 = model_propcess(inputs_start, adj, v0)
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
                # print("tmp", tmp.size())
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

    for disease_name in disease_list:
        val_dir = data_path + f"/{disease_name}" + f"/{disease_name}"
        test_dataset = My_dHCP_Data_tau(val_dir)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        #                                           sampler=test_sampler)
        # -------------------------------------------------------------------------------
        mae, mae_list = test(test_loader)
        with open(test_result_path + "/" + "mae.txt", "a") as f:
            f.write(str(disease_name) + ',' + str(mae) + "\n")
        f.close()
        mae_all += mae * len(mae_list)
        num += len(mae_list)
    mae_all = mae_all / num
    with open(test_result_path + "/" + "mae.txt", "a") as f:
        f.write(str("MAE") + ',' + str(mae_all) + "\n")
    f.close()
    print('finish')







