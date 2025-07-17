#Implementando uma Deep Learning do zero em py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
#import keras
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transforms = transforms.ToTensor()
#definindo a conversão de imagem para tensor

trainset = datasets.MNIST('/MNIST_data/', download= True, train=True, transform=transforms)
#Carrega a parte de treino do dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size= 64, shuffle=True)
#Cria um buffer para pegar os dados por partes

valset = datasets.MNIST('/MNIST_data/', download= True, train=False, transform=transforms)
#Carrega a parte de validação do datasets
valloader = torch.utils.data.DataLoader(valset, batch_size= 64, shuffle=True)
#cria um buffer para pegar os dados por partes

dataiter = iter(trainloader)
imagens, etiquetas = next(dataiter)

# Convertendo corretamente para visualização
img = imagens[0].cpu().numpy().squeeze()
plt.imshow(img, cmap='gray_r')
plt.title(f'Etiqueta: {etiquetas[0].item()}')
plt.axis('off')
plt.show()


#print(imagens[0].shape)#Para verificar as dimensões do tensor de cada imagem 
#print(etiquetas[0].shape)#para verificar as dimensoes do tensor de cada etiqueta

#keras.applications.InceptionV3(
#    include_top=True,
#    weights="imagenet",
#    input_tensor=None,
#    input_shape=None,
#    pooling=None,
#    classes=1000,
#    classifier_activation="softmax",
#    name="inception_v3",
#)

class modelo(nn.Module):
    def __init__(self):
        super(modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)#camada de entrada, 784 neuronios que se ligam a 128
        self.linear2 = nn.Linear(128, 64)#camada interna 1, 128 neuronios que se ligam a 64
        self.linear3 = nn.Linear(64, 10)#camada interna 2 ,64 neuronios que se ligam a 10
        #para a camada de saída não é necessário definir nada pois só precisamos pegar o output da camada interna 2
        
    def forward(self, x):
        x = F.relu(self.linear1(x))  #função de ativação da camada de entrada para a camada interna 1 
        x = F.relu(self.linear2(x))  #função de ativação da camada 1 para a camada interna 2
        x = self.linear3(x) #função de ativação da camada interna 2 para a camada de saída, nesse caso f(x) = x
        return F.log_softmax(x, dim=1)   #dados utillizados para calcular a perda
    
def treino(modelo, trainloader, device):
     
    otimizador = optim.SGD(modelo.parameters(), lr=0.01,momentum=0.5)#define a politica de atualização dos pesos e da bias 
    inicio = time()#timer para sabermos quanto tempo levou o treino
    
    criterio = nn.NLLLoss()#definindo o criterio para calcular a perda
    EPOCHS = 10 #numero de epochs que o algoritmo rodará
    modelo.train() #ativando o modo de treinamento do modelo 
    
    for epoch in range(EPOCHS):
        perda_acumulada = 0 #inicialização da perda acumulada da epoch em questão 
        
        for imagens, etiquetas in trainloader:
            
            imagens = imagens.view(imagens.shape[0], -1 ) #convertendo as imagens para "vetores" de 28*28 casas para ficar compativeis
            otimizador.zero_grad() #zerando os gradientes por conta do ciclo anterior 
            
            output = modelo(imagens.to(device))#colocando os dados no modelo 
            perda_instantanea = criterio(output, etiquetas.to(device))#calculando as perdas da epoch em questão 
            
            perda_instantanea.backward() #back propagation a partir da perda
            
            otimizador.step()#atualizando os pesos e a bias 
            
            perda_acumulada += perda_instantanea.item()
        
        
        else:
            print("Epoch {} - Perda resultante : {}".format(epoch+1, perda_acumulada/len(trainloader)))
    print("\nTempo de treino (em minutos) = ", (time()-inicio)/60) 

def validacao(modelo, valloader, device):
    conta_corretas, conta_todas = 0, 0
    for imagens, etiquetas in valloader:
        for i in range(len(etiquetas)):
            img = imagens[i].view(1,784)
            # desativar o autograd para acelerar a validação 
            with torch.no_grad():
                logps = modelo(img.to(device))#output do modelo em escala logaritmica 
                    
            ps = torch.exp(logps) #converte output para a escala normal
            probab = list(ps.cpu().numpy()[0])
            etiqueta_pred = probab.index(max(probab)) #converte o tensor em numérico
            etiqueta_certa = etiquetas.numpy()[i] 
            if (etiqueta_certa == etiqueta_pred): #compara a previsão com o valor correto
                conta_corretas += 1
            conta_todas += 1
            
    print(f'Total de imagens testadas = {conta_todas}')            
    print('\nPrecisão do Modelo = {}%'.format(conta_corretas*100/conta_todas)) 

modelo=modelo()
device= torch.device("cuda" if torch.cuda.is_available() else "cpu") 
modelo.to(device)        