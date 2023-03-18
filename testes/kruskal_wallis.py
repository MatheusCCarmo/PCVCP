from scipy.stats import kruskal

# Kruskal-Wallis
statistic, pvalue = kruskal()

# resultado 
print("Estatística de teste: {:.2f}".format(statistic))
print("Valor p: {:.4f}".format(pvalue))
