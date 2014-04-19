## Fitting robust and standard covariance to party location data
## @griverorz
## Sat Apr 19 14:41:56 PDT 2014

import os
import pandas as pd
import pylab as pl
import random
from sklearn.covariance import EmpiricalCovariance, MinCovDet

## Copying data from the locparties project
os.chdir("/Users/gonzalorivero/Documents/datablog/outliers")
outliers = pd.read_csv("./dta/outliers.csv")

## Add some jitter to the data for viz purposes
jitter = .5
outliers.Mloc_ideo = [h + random.uniform(-jitter, jitter)
    for h in outliers.Mloc_ideo]

outliers.Mloc_nacl = [h + random.uniform(-jitter, jitter)
    for h in outliers.Mloc_nacl]

## Distribution of parties in the political arena
for i in outliers.party.unique():
    pl.plot(outliers.Mloc_nacl[outliers['party'] == i],
        outliers.Mloc_ideo[outliers['party'] == i], 'o')
    pl.xlim(1, 10)
    pl.ylim(1, 10)
    pl.show()

pl.savefig("./img/dist.png")
pl.close()

## Remove NA data (they are probably the most likely outliers (!?))
outliers = outliers.dropna()

## Fit only the PP
X = outliers[outliers["party"] == 1][["Mloc_nacl", "Mloc_ideo"]].values
## Robust covariance
robust_cov = MinCovDet().fit(X)
## MLE covariance
emp_cov = EmpiricalCovariance().fit(X)

pl.scatter(X[:, 0], X[:, 1], color='blue', alpha=0.15)
pl.xlim(1, 10)
pl.ylim(1, 10)

## Normalize space to be plotted
xx, yy = np.meshgrid(np.linspace(1, 10, 10),
                     np.linspace(1, 10, 10))
zz = np.c_[xx.ravel(), yy.ravel()]

## Select the levels to be plotted
levels = [0., 1., 2., 3., 4., 5.]

## Boy, I hate matplotlib
mahal_robust_cov = robust_cov.mahalanobis(zz)
mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
robust_contour = pl.contour(xx, yy, np.sqrt(mahal_robust_cov), levels,
    linestyles='-', cmap=pl.cm.YlOrBr_r, linewidths=2.)
pl.clabel(robust_contour, inline=1, fontsize=10)

mahal_emp_cov = emp_cov.mahalanobis(zz)
mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
mahal_contour = pl.contour(xx, yy, np.sqrt(mahal_emp_cov), levels,
    linestyles='--', cmap=pl.cm.PuBu_r, linewidths=2.)
pl.clabel(mahal_contour, inline=1, fontsize=10)

pl.legend([mahal_contour.collections[1], robust_contour.collections[1]],
    ['MLE', 'Robust'], loc="upper left", borderaxespad=0)
pl.savefig("./img/outliers.png")
pl.close()
