#primeiro código de ML do BOOTCAMP
import matplotlib.pyplot as plt
#Importa o módulo pyplot da biblioteca Matplotlib com o apelido plt
from sklearn.datasets import make_regression
#Importa a função make_regression(gera dados sintéticos ,ou seja, falsos) da biblioteca scikit-learn.

x, y = make_regression(n_samples=200, n_features=1, noise=30)
#Cria um conjunto de dados com 200 amostras (n_samples=200) e 1 única variável explicativa (n_features=1).
#O parâmetro noise=30 adiciona ruído aleatório nos dados, ou seja, torna os pontos mais espalhados, simulando dados reais (com imperfeições).
#x: é uma matriz 200x1 com os valores das variáveis de entrada.
#y: é um vetor com 200 valores, representando a variável dependente (alvo) que depende de x.

plt.scatter(x,y)
#Cria um gráfico de dispersão (scatter plot) com os valores de x no eixo X e y no eixo Y.

plt.show()
#Exibe o gráfico na tela.

#Sobre o NOISE
#Sem ruído (noise=0), os pontos estariam perfeitamente alinhados numa reta.
#Com ruído (noise=30), a função adiciona um valor aleatório à variável y, causando variação e dispersão nos pontos.
#Isso simula o que acontece na vida real: os dados geralmente têm imperfeições, medições imprecisas, erros experimentais etc.
#Exemplo visual:
#Sem ruído → todos os pontos x, y ficam exatamente sobre uma linha reta.
#Com ruído → os pontos ficam ao redor da linha, parecendo uma nuvem.