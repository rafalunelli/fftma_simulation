# precisa as seguintes biblios:
#https://mmaelicke.github.io/scikit-gstat/install.html
#https://pythonhosted.org/scikit-fuzzy/install.html

from FFTMA import *
import seaborn as sns
import matplotlib.pyplot as plt
import skgstat as skg

m = 500
n = 500
range_h = 20
range_v = 20
err = 1
model = 3
output = fftma_l3c(m,n, range_h, range_v,err,model)

####plot
sns.heatmap(output,cmap="YlGnBu")

#histogram
plt.hist(output.reshape(m*n), bins=150)

####variogram
#values = output.reshape(m*n)
#coordinates = np.arange(0,m*m)
#V = skg.Variogram(coordinates, values,n_lags=25,maxlag=60)
#V.plot()