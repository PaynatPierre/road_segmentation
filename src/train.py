from config import cfg
from model import get_unet, get_unet_pretrained, get_unet_complete_pretrained, get_discriminator
import torch
from dataloader import get_dataloader
import tqdm
import statistics
import os
from loss_and_metrics import IoULoss, IoU_metric
import numpy as np
from PIL import Image
from postprocess import post_process


'''
this function is made to manage all the training pipeline, including validation. It also save models checkpoint after each validation.

args:
    None

return:
    None

'''
def train():
    torch.manual_seed(0)

    # loading model, loss and optimizer
    model = get_unet_complete_pretrained().cuda()
    
    # criterion = IoULoss().cuda()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    # create dataloader
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader()

    # create list to deal with all results and save them
    train_list_loss = []
    train_list_acc = []
    val_list_loss = []
    val_list_acc = []
    val_list_acc_post_process = []
    loss_history = []
    acc_history = []
    acc_post_process_history = []

    # index of checkpoint and bath counting creation
    validation_count = 0
    checkpoint_count = 0
    batch_after_last_validation = 0
    for epoch in range(cfg.TRAIN.NBR_EPOCH):
        
        # creation of index to count gradian accumulation since the last weights update
        gradiant_accumulation_count = 0

        # loop en batchs
        train_range = tqdm.tqdm(train_dataloader)
        for i, (datas, labels) in enumerate(train_range):

            # change model mode
            model.train()
            gradiant_accumulation_count += 1
            batch_after_last_validation += 1

            # moove variables place in memory on GPU RAM
            if torch.cuda.is_available():
                datas = datas.cuda()
                labels = labels.cuda()

            # make prediction
            outputs = model(datas)

            # compute loss
            loss = criterion(outputs, labels)

            # if torch.cuda.is_available():
            #     loss.cuda()

            # make gradiant retropropagation
            loss.backward()

            # condition to choose if you update model's weights or not
            if gradiant_accumulation_count >= cfg.TRAIN.GRADIANT_ACCUMULATION or i == len(train_dataloader) - 1:

                # reinitialisation of gradiant accumulation index
                gradiant_accumulation_count = 0

                # update model's weights
                optimizer.step()
                optimizer.zero_grad()

                # compute mectric
                metric = IoU_metric(outputs, labels)

                # save loss and metric for the current batch
                train_list_loss.append(loss.item())
                train_list_acc.append(metric.item())

                # update tqdm line information
                train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f || metric: %4.4f" % (epoch, statistics.mean(train_list_loss), statistics.mean(train_list_acc)))
                train_range.refresh()

            # condition to choose if you have to do a validation or not
            if batch_after_last_validation + 1 > len(train_dataloader)/cfg.TRAIN.VALIDATION_RATIO:

                validation_count += 1

                # remove gradiants computation for the validation
                with torch.no_grad():
                    batch_after_last_validation = 0

                    # validation loop
                    for j, (val_datas, val_labels) in enumerate(validation_dataloader):
                        
                        # change model mode
                        model.eval()

                        # moove variables place in memory on GPU RAM
                        if torch.cuda.is_available():
                            val_datas = val_datas.cuda()
                            val_labels = val_labels.cuda()
                        
                        # make prediction
                        outputs = model(val_datas)
                        postprocessed_outputs = post_process(outputs)

                        if not os.path.isdir(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation')):
                            os.mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation'))

                        for k in range(outputs.shape[0]):
                            img = torch.argmax(outputs[k], dim=0)
                            img = torch.unsqueeze(img, dim=-1)
                            img = torch.cat([img,img,img], dim=-1)

                            img = img*255
                            img = img.cpu().detach().numpy()
                            img = Image.fromarray(np.uint8(img))

                            img.save(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation/img_pred_val_{validation_count}_sample_{k}.png'))


                            img_2 = torch.argmax(val_labels[k], dim=0)
                            img_2 = torch.unsqueeze(img_2, dim=-1)
                            img_2 = torch.cat([img_2,img_2,img_2], dim=-1)

                            img_2 = img_2*255
                            img_2 = img_2.cpu().detach().numpy()
                            img_2 = Image.fromarray(np.uint8(img_2))

                            img_2.save(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation/label_val_{validation_count}_sample_{k}.png'))


                            img_3 = torch.argmax(postprocessed_outputs[k], dim=0)
                            img_3 = torch.unsqueeze(img_3, dim=-1)
                            img_3 = torch.cat([img_3,img_3,img_3], dim=-1)

                            img_3 = img_3*255
                            img_3 = img_3.cpu().detach().numpy()
                            img_3 = Image.fromarray(np.uint8(img_3))

                            img_3.save(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation/img_pred_postprocessed_val_{validation_count}_sample_{k}.png'))


                        # compute loss
                        loss = criterion(outputs, val_labels)

                        # compute mectric
                        metric = IoU_metric(outputs, val_labels)
                        post_process_metric = IoU_metric(postprocessed_outputs, val_labels)

                        # save loss and metric for the current batch
                        val_list_loss.append(loss.item())
                        val_list_acc.append(metric.item())
                        val_list_acc_post_process.append(post_process_metric.item())

                    # print validation results
                    print(" ")
                    print("VALIDATION -> epoch: %4d || loss: %4.4f || metric: %4.4f || post processed metric: %4.4f" % (epoch, statistics.mean(val_list_loss), statistics.mean(val_list_acc), statistics.mean(val_list_acc_post_process)))

                    # save model checkpoint
                    torch.save(model.state_dict(), os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH,'ckpt_' + str(checkpoint_count)) + "_metric_" + str(round(statistics.mean(val_list_acc),5)) + ".ckpt")
                    checkpoint_count += 1
                    print(" ")

                    # save loss and metric for the current epoch
                    loss_history.append(statistics.mean(val_list_loss))
                    acc_history.append(statistics.mean(val_list_acc))
                    acc_post_process_history.append(statistics.mean(val_list_acc_post_process))

                    with open(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, 'result_history.txt'), 'a+') as result_file:
                        result_file.write(f"checkpoint_{checkpoint_count} : loss = {statistics.mean(val_list_loss)} , metric = {statistics.mean(val_list_acc)} , postprocessed_metric = {statistics.mean(val_list_acc_post_process)} \n")


                    # print loss and metric history
                    print("loss history:")
                    print(loss_history)
                    print("acc history:")
                    print(acc_history)
                    print("acc post processed history:")
                    print(acc_post_process_history)

                    # clear storage of short term result
                    train_list_loss = []
                    train_list_acc = []
                    val_list_loss = []
                    val_list_acc = []
                    val_list_acc_post_process = []
        
        # clear storage of short term result
        train_list_loss = []
        train_list_acc = []




'''
this function is made to manage all the training pipeline, including validation. It also save models checkpoint after each validation.

args:
    None

return:
    None

'''
def train_with_discriminateur():
    torch.manual_seed(0)

    # loading model, loss and optimizer
    model = get_unet().cuda()
    discriminateur = get_discriminator().cuda()
    
    # criterion = IoULoss().cuda()
    criterion_unet = torch.nn.BCELoss()
    criterion_disc = torch.nn.BCELoss()
    optimizer_unet = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    optimizer_disc = torch.optim.Adam(discriminateur.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    # create dataloader
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader()

    # create list to deal with all results and save them
    train_list_loss = []
    train_list_acc = []
    train_list_acc_disc = []
    val_list_loss = []
    val_list_acc = []
    val_list_acc_post_process = []
    loss_history = []
    acc_history = []
    acc_post_process_history = []

    # index of checkpoint and bath counting creation
    validation_count = 0
    checkpoint_count = 0
    batch_after_last_validation = 0
    for epoch in range(cfg.TRAIN.NBR_EPOCH):
        
        # creation of index to count gradian accumulation since the last weights update
        gradiant_accumulation_count = 0

        # loop en batchs
        train_range = tqdm.tqdm(train_dataloader)
        for i, (datas, labels) in enumerate(train_range):

            # change model mode
            model.train()
            discriminateur.train()

            gradiant_accumulation_count += 1
            batch_after_last_validation += 1

            # moove variables place in memory on GPU RAM
            if torch.cuda.is_available():
                datas = datas.cuda()
                labels = labels.cuda()

            outputs_disc = discriminateur(labels)
            tmp_1 = torch.ones((labels.shape[0],1)).cuda()
            tmp_2 = torch.zeros((labels.shape[0],1)).cuda()

            loss_dict_real = criterion_disc(outputs_disc, torch.cat((tmp_1, tmp_2), dim=1)).cuda()
            loss_dict_real.backward(retain_graph=True)

            # make prediction
            outputs = model(datas)
            outputs_disc_2 = discriminateur(outputs)
            
            # compute loss
            loss = criterion_unet(outputs, labels) + criterion_unet(outputs_disc_2,  torch.cat((tmp_1, tmp_2), dim=1))
            loss.backward(retain_graph=True)

            loss_dict_fake =  criterion_disc(outputs_disc_2,  torch.cat((tmp_2, tmp_1), dim=1))
            loss_dict_fake.backward(retain_graph=True)
            # loss_dict = criterion_disc(outputs_disc, torch.cat((tmp_1, tmp_2), dim=1)).cuda() + criterion_disc(outputs_disc_2,  torch.cat((tmp_2, tmp_1), dim=1))
            

            # # make gradiant retropropagation
            # loss.backward()
            # print('chameau')
            # loss_dict.backward()

            # condition to choose if you update model's weights or not
            if gradiant_accumulation_count >= cfg.TRAIN.GRADIANT_ACCUMULATION or i == len(train_dataloader) - 1:

                # reinitialisation of gradiant accumulation index
                gradiant_accumulation_count = 0

                optimizer_disc.step()
                optimizer_disc.zero_grad()

                # update model's weights
                optimizer_unet.step()
                optimizer_unet.zero_grad()

                # compute mectric
                metric = IoU_metric(outputs, labels)
                metric_disc = (torch.sum(torch.argmin(outputs_disc, dim=1)).item() + torch.sum(torch.argmax(outputs_disc_2, dim=1)).item())/(2*labels.shape[0])

                # save loss and metric for the current batch
                train_list_loss.append(loss.item())
                train_list_acc.append(metric.item())
                train_list_acc_disc.append(metric_disc)

                # update tqdm line information
                train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f || metric: %4.4f || metric_disc: %4.4f" % (epoch, statistics.mean(train_list_loss), statistics.mean(train_list_acc), statistics.mean(train_list_acc_disc)))
                train_range.refresh()

            # condition to choose if you have to do a validation or not
            if batch_after_last_validation + 1 > len(train_dataloader)/cfg.TRAIN.VALIDATION_RATIO:

                validation_count += 1

                # remove gradiants computation for the validation
                with torch.no_grad():
                    batch_after_last_validation = 0

                    # validation loop
                    for j, (val_datas, val_labels) in enumerate(validation_dataloader):
                        
                        # change model mode
                        model.eval()

                        # moove variables place in memory on GPU RAM
                        if torch.cuda.is_available():
                            val_datas = val_datas.cuda()
                            val_labels = val_labels.cuda()
                        
                        # make prediction
                        outputs = model(val_datas)
                        postprocessed_outputs = post_process(outputs)

                        if not os.path.isdir(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation')):
                            os.mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation'))

                        for k in range(outputs.shape[0]):
                            img = torch.argmax(outputs[k], dim=0)
                            img = torch.unsqueeze(img, dim=-1)
                            img = torch.cat([img,img,img], dim=-1)

                            img = img*255
                            img = img.cpu().detach().numpy()
                            img = Image.fromarray(np.uint8(img))

                            img.save(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation/img_pred_val_{validation_count}_sample_{k}.png'))


                            img_2 = torch.argmax(val_labels[k], dim=0)
                            img_2 = torch.unsqueeze(img_2, dim=-1)
                            img_2 = torch.cat([img_2,img_2,img_2], dim=-1)

                            img_2 = img_2*255
                            img_2 = img_2.cpu().detach().numpy()
                            img_2 = Image.fromarray(np.uint8(img_2))

                            img_2.save(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation/label_val_{validation_count}_sample_{k}.png'))


                            img_3 = torch.argmax(postprocessed_outputs[k], dim=0)
                            img_3 = torch.unsqueeze(img_3, dim=-1)
                            img_3 = torch.cat([img_3,img_3,img_3], dim=-1)

                            img_3 = img_3*255
                            img_3 = img_3.cpu().detach().numpy()
                            img_3 = Image.fromarray(np.uint8(img_3))

                            img_3.save(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, f'validation_visualisation/img_pred_postprocessed_val_{validation_count}_sample_{k}.png'))


                        # compute loss
                        loss = criterion_unet(outputs, val_labels)

                        # compute mectric
                        metric = IoU_metric(outputs, val_labels)
                        post_process_metric = IoU_metric(postprocessed_outputs, val_labels)

                        # save loss and metric for the current batch
                        val_list_loss.append(loss.item())
                        val_list_acc.append(metric.item())
                        val_list_acc_post_process.append(post_process_metric.item())

                    # print validation results
                    print(" ")
                    print("VALIDATION -> epoch: %4d || loss: %4.4f || metric: %4.4f || post processed metric: %4.4f" % (epoch, statistics.mean(val_list_loss), statistics.mean(val_list_acc), statistics.mean(val_list_acc_post_process)))

                    # save model checkpoint
                    torch.save(model.state_dict(), os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH,'ckpt_' + str(checkpoint_count)) + "_metric_" + str(round(statistics.mean(val_list_acc),5)) + ".ckpt")
                    checkpoint_count += 1
                    print(" ")

                    # save loss and metric for the current epoch
                    loss_history.append(statistics.mean(val_list_loss))
                    acc_history.append(statistics.mean(val_list_acc))
                    acc_post_process_history.append(statistics.mean(val_list_acc_post_process))

                    with open(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, 'result_history.txt'), 'a+') as result_file:
                        result_file.write(f"checkpoint_{checkpoint_count} : loss = {statistics.mean(val_list_loss)} , metric = {statistics.mean(val_list_acc)} , postprocessed_metric = {statistics.mean(val_list_acc_post_process)} \n")


                    # print loss and metric history
                    print("loss history:")
                    print(loss_history)
                    print("acc history:")
                    print(acc_history)
                    print("acc post processed history:")
                    print(acc_post_process_history)

                    # clear storage of short term result
                    train_list_loss = []
                    train_list_acc = []
                    val_list_loss = []
                    val_list_acc = []
                    val_list_acc_post_process = []
        
        # clear storage of short term result
        train_list_loss = []
        train_list_acc = []