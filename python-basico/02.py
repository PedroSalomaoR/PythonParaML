#tipos de variaveis para trabalhar com ML
int = 7
float = 7.5
str = "Olá, Mundo!"
bool = True or False

#COMPLEX
# Criando um número complexo de duas formas
z1 = complex(2, 3)      # 2 + 3j
z2 = 2 + 3j             # também 2 + 3j

# Acessando parte real e imaginária
print(z1.real)          # 2.0
print(z1.imag)          # 3.0

# Conjugado (troca o sinal da parte imaginária)
print(z1.conjugate())   # 2 - 3j

# Operações com complexos
z3 = 1 + 2j
z4 = 2 - 1j

print(z3 + z4)          # 3 + 1j
print(z3 * z4)          # (1+2j)*(2-1j) = 4 + 3j
