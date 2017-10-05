N = Int(5e1)  # number of data
M = Int(1e5)  # number of realisations

mu = [7 10] * 1.0  # mean of data
c = [5 -3; -3 7] * 1.0  # covarianza en float
s =

# generate data
# Pkg.add("PDMats")
using Distributions
probMVgaus = MvNormal(c)
x = reshape(rand(probMVgaus,N*M), 2, N, M)

# estimo los mu y c de cada relizacion
muest = mean(x, 2)
dif = x - muest # ver como hacer esta resta facilmente
cest = np.sum(dif.reshape((N,M,2,1)) * dif.reshape((N,M,1,2)), axis=0) / (N - 1)


# saco la media y varianza entre todas las realizaciones
muExp = np.mean(muest, axis=0)
difmu = muest - muExp  # aca tambien esta bien restado
muVar = np.sum(difmu.reshape((M,2,1)) * difmu.reshape((M,1,2)), axis=0) / (M - 1)

cExp = np.mean(cest, axis=0)
difc = cest - cExp ## ACA ESTA EL PROBLEMA
cVar = np.sum(difc.reshape((M,2,2,1,1)) *
               difc.reshape((M,1,1,2,2)).transpose((0,1,2,4,3)),
               axis=0) / (M - 1)

#cVar2 = np.zeros((M,2,2,2,2))
#for i in range(M):
#    cVar2[i] = difc[i].reshape((2,2,1,1)) * difc[i].reshape((1,1,2,2))
#
#cVar2 = np.sum(cVar2 / (M - 1), axis=0)
#
#cVar2 = np.zeros_like(cVar)
#for i in [0,1]:
#    for j in [0,1]:
#        for k in [0,1]:
#            for l in [0,1]:
#                cVar2[i,j,k,l] = np.sum(difc[:,i,j] * difc[:,k,l])

#difc2 = difc.reshape((-1,4))
#cVar2 = np.sum(difc2.reshape((-1,4,1)) * difc2.reshape((-1,1,4)) / (M - 1),
#               axis=0)
#cVar2 = cVar2.reshape((2,2,2,2))



# saco las cuentas analiticas
expMu = mu
VarMu = c / N
expC = c
# no es necesario trasponer porque c es simetrica
VarC = (c.reshape((2,2,1,1)) *
        c.reshape((1,1,2,2)).transpose((0,1,3,2))) * (2 * N - 1) / (N - 1)**2
#VarC2 = c.reshape((4,1)) * c.reshape((1,4)) * (2 * N - 1) / (N - 1)**2
#VarC2 = VarC2.reshape((2,2,2,2))


print('numerico: \n', cVar, '\n\n\n\n analitico \n', VarC)

#reldif = np.abs((cVar - VarC) / VarC)
#reldif > 1e-1
