N = Int(5e1)  # number of data
M = Int(1e5)  # number of realisations

mu = [7; 10] * 1.0  # mean of data
c = [5 3; 3 7] * 1.0  # covarianza en float

# generate data
using Distributions
probMVgaus = MvNormal(mu,c)
x = rand(probMVgaus, N*M)
x = reshape(x, N, M, 2)

# estimo los mu y c de cada relizacion
muest = mean(x, 1)
dif = reshape(x .- muest,N,M,2,1) # ver como hacer esta resta facilmente
cest = sum(dif .* permutedims(dif,[1,2,4,3]), 1) / (N - 1)

# saco la media y varianza de los estimadores entre todas las realizaciones
# media y var de mu
muExp = mean(muest, 2)
difmu = reshape(muest .- muExp, M, 2, 1)
muVar = sum(difmu .* permutedims(difmu,[1,3,2]), 1) / (M - 1)
# media y var de c
cExp = mean(cest, 2)
difc = reshape(cest .- cExp, M,2,2,1,1)
cVar = sum(difc .* permutedims(difc, [1,5,4,3,2]), 1) / (M - 1)
cVar = cVar[1,:,:,:,:]

# saco las cuentas analiticas
expMu = mu
VarMu = c / N
expC = c
# no es necesario trasponer porque c es simetrica
cShaped = reshape(c,2,2,1,1)
VarCnn = cShaped .* permutedims(cShaped, [4,3,2,1])
nCons = (2 * N - 1) / (N - 1)^2
VarC = VarCnn * nCons

print("numerico: \n", cVar, "\n analitico \n", VarC)
print(cVar / nCons)
print(VarCnn)
