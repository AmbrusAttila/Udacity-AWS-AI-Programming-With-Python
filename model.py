import torch
from torchvision import models
from torch import nn, optim
from collections import OrderedDict

from dataloader import process_image

def create_model(arch, hidden_units):

    model=None
    input_units=0

    if arch=="vgg13":
        model=models.vgg13(pretrained=True)
        input_units=25088
    elif arch=="resnet50":
        model=models.resnet50(pretrained=True)
        input_units=2048
    elif arch=="densenet121":
        model=models.densenet121(pretrained=True)
        input_units=1024 

    classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(input_units, hidden_units)),
            ('relu', nn.ReLU()),
            ('drp', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ])
    )

    for param in model.parameters():
        param.requires_grad = False

    if(arch=="resnet50"):
        model.fc=classifier
    else:
        model.classifier=classifier

    return model

def train_model(epochs, trainloader, validloader, device, model, optimizer, criterion):
    PRINT_N=5

    total_train_loss=0
    train_accuracy=0
    steps=0
    epoch=0

    for epoch in range(epochs):
        for imgs, labs in trainloader:
            steps+=1
            imgs, labs=imgs.to(device), labs.to(device)
        
            optimizer.zero_grad() 
        
            tlogps=model.forward(imgs)
            tloss=criterion(tlogps, labs)        
            tloss.backward()
        
            optimizer.step()
        
            total_train_loss+=tloss.item()
        
            tps=torch.exp(tlogps)
            top_tps, top_tclass=tps.topk(1, dim=1)
            tequals = top_tclass == labs.view(*top_tclass.shape)
            train_accuracy += torch.mean(tequals.type(torch.FloatTensor)).item()
        
            if steps % PRINT_N==0:
            
                total_valid_loss=0
                valid_accuracy=0
            
                model.eval()
            
                with torch.no_grad():
                    for vimgs, vlabs in validloader:
                        vimgs, vlabs=vimgs.to(device), vlabs.to(device)
                
                        vlogps=model.forward(vimgs)
                        vloss=criterion(vlogps, vlabs) 
                        total_valid_loss+=vloss.item()
                
                        vps=torch.exp(vlogps)
                        top_vps, top_vclass=vps.topk(1, dim=1)
                        vequals = top_vclass == vlabs.view(*top_vclass.shape)
                        valid_accuracy += torch.mean(vequals.type(torch.FloatTensor)).item()

                div=len(validloader)
                print(f"Epoch: {epoch+1}/{epochs} Steps: {steps}")
                print(f"Train Loss: {total_train_loss/PRINT_N:.3f} Valid Loss: {total_valid_loss/div:.3f}")
                print(f"Train Accuracy: {train_accuracy/PRINT_N:.3f} Valid Accuracy: {valid_accuracy/div:.3f}")
                total_train_loss=0
                train_accuracy=0
                model.train()
    return epoch, steps

def save_checkpoint(filepath, model, optimizer, epoch, step, arch, hidden_units, learning_rate):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'lr': learning_rate,
        'trained_layer': model.fc if arch=="resnet50" else model.classifier,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'steps': step
    }

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    lr = checkpoint['lr']

    model = create_model(arch,hidden_units) 

    if arch=="resnet50":
        model.fc=checkpoint['trained_layer']
    else:
        model.classifier=checkpoint['trained_layer'] 
     
    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    optimizer = optim.Adam(
        model.fc.parameters() if arch=="resnet50" else model.classifier.parameters(),
        lr=lr
    )

    optimizer.load_state_dict(
         checkpoint['optimizer_state_dict']
    )

    epoch=checkpoint['epoch']
    steps=checkpoint['steps'] 
    
    return model, optimizer, epoch, steps

def predict_model(image_path, device, model, top_k):
    img = process_image(image_path)
    img_tens=torch.tensor(img).float().unsqueeze(0).to(device)
    
    model.eval()
    
    with torch.no_grad():
        logps=model.forward(img_tens)
        ps=torch.exp(logps)
        top_pss, top_classes=ps.topk(top_k)
        
    return top_pss, top_classes
