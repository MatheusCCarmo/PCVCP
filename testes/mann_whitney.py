from scipy.stats import mannwhitneyu

# Mann-Whitney
statistic, pvalue = mannwhitneyu()

# Resultado
print("Estatística de teste: {:.2f}".format(statistic))
print("Valor p: {:.4f}".format(pvalue))
