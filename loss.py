import torch
import torch.nn.functional as F


def KL_dist(vector1, vector2):
    vector1 = F.softmax(vector1, dim=0)
    vector2 = F.softmax(vector2, dim=0)
    dist = F.kl_div(vector1.log(), vector2, reduction='sum')
    return dist


def L1_dist(vector1, vector2):
    size = vector1.size()[0]
    vector1 = torch.sigmoid(vector1)
    vector2 = torch.sigmoid(vector2)
    dist = torch.dist(vector1, vector2, p=1)
    dist = dist / size
    return dist


def L1_ratio(vector1, vector2):
    dist = torch.dist(vector1, vector2, p=1)
    dist = dist / (torch.norm(vector1, p=1) + torch.norm(vector2, p=1))
    return dist


def calculate_dist(vector1, vector2):
    vector1 = torch.sigmoid(vector1)
    vector2 = torch.sigmoid(vector2)
    out = torch.dist(vector1, vector2)
    return out


def KL_loss(outputs, targets):
    batchsize = outputs.size(0)
    assert batchsize == targets.size(0)

    count = 0
    distance = 0.0
    for i in range(batchsize):
        tar = targets[i]
        for j in range(i+1, batchsize):
            if (targets[j] == tar):
                distance += KL_dist(outputs[i], outputs[j])
                count += 1
    if count == 0:
        count = 1
    loss = distance / count
    return loss


def L1_loss(outputs, targets, mode='dist'):
    batchsize = outputs.size(0)
    assert batchsize == targets.size(0)

    count = 0
    distance = 0.0
    for i in range(batchsize):
        tar = targets[i]
        for j in range(i+1, batchsize):
            if (targets[j] == tar):
                if(mode == 'dist'):
                    distance += L1_dist(outputs[i], outputs[j])
                elif(mode == 'ratio'):
                    distance += L1_ratio(outputs[i], outputs[j])
                else:
                    raise ValueError('\'mode\' should be either \'dist\' or \'ratio\'!')
                count += 1
    loss = distance / count
    return loss


def ratio_loss(outputs, targets):
    batchsize = outputs.size(0)
    assert batchsize == targets.size(0)

    embedding_shape = outputs.size()[1:]
    feature_count = [0, 0, 0]
    loss_count = [0, 0, 0]
    distance = [0.0, 0.0, 0.0]
    sum = torch.zeros(3, embedding_shape[0])
    sum = sum.to('cuda')
    for i in range(batchsize):
        tar = targets[i]
        feature_count[tar] = feature_count[tar] + 1
        sum[tar] = sum[tar] + outputs[i]
        for j in range(i+1, batchsize):
            if (targets[j] == tar):
                loss_count[tar] = loss_count[tar] + 1
                distance[tar] = distance[tar] + calculate_dist(outputs[i], outputs[j])

    for i in range(3):
        if (feature_count[i] == 0):
            feature_count[i] = 1
        if (loss_count[i] == 0):
            loss_count[i] = 1

    inner_distance = distance[0] / loss_count[0] + distance[1] / loss_count[1] + distance[2] / loss_count[2]
    average = [sum[0] / feature_count[0], sum[1] / feature_count[1], sum[2] / feature_count[2]]
    inter_distance = calculate_dist(average[0], average[1]) + calculate_dist(average[0], average[2]) + calculate_dist(average[1], average[2])
    loss = inner_distance / inter_distance
    return loss
