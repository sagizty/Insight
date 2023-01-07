import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix


def data_import(src_path):
    """
    :param src_path:
    :return:
    x：dataframe,(n_samples, n_features)
    y：numpy,(n_samples, label)
    """
    dataset = pd.read_csv(src_path, sep=",")
    # 提取label
    y = dataset['label'].values
    x = dataset.drop(columns=['label'])
    return x, y


def plot_clustering_cm(y, clu_pred, lantent_dimension=16):
    # Evaluate the performance of the model
    ari = adjusted_rand_score(y, clu_pred)
    ami = adjusted_mutual_info_score(y, clu_pred)

    # Create a confusion matrix
    matrix = confusion_matrix(y, clu_pred)

    # Visualize the confusion matrix using a heatmap
    sns.heatmap(matrix, annot=True, fmt='d', cmap='PuBu', annot_kws={'size': 10})

    plt.ylabel('True val')
    plt.xlabel('Pred cluster')

    xlocations = np.arange(10)
    start = ord('a') - 1

    plt.xticks(xlocations, [chr(start + i) for i in range(1, 11)])
    plt.yticks(xlocations, [i for i in range(1, 11)])
    plt.ylim(10)

    plt.title(f'Adjusted Rand Index: {ari:.3f}' + f'\nAdjusted Mutual Information: {ami:.3f}')
    plt.savefig('./clustering_cm of lantent dimension ' + str(lantent_dimension) + '.jpg', dpi=300)
    plt.show()


def data_plot_2D(src_path, epoch):
    df = pd.read_csv(src_path, sep=",")
    df.columns = ['label', 'x', 'y']
    fig, ax = plt.subplots()
    groups = df.groupby('label')
    for type, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=2, label=type)
    ax.legend(loc='right')
    ax.set_title('2D feature visualization at epoch ' + str(epoch))
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.savefig('./2D_plot at epoch' + str(epoch) + '.jpg', dpi=300)
    plt.show()


# data_plot_all_2D
for epoch in range (1,11):
    src_path = r'./imaging_results/2/latent_2_Epoch_' + str(epoch) + '.csv'
    data_plot_2D(src_path, epoch)

src_path = r'./imaging_results/2/latent_2_Epoch_10.csv'
x, y = data_import(src_path)
cluster = KMeans(n_clusters=10, random_state=42).fit(x)
# pred for record train
clu_pred = cluster.predict(x)
plot_clustering_cm(y, clu_pred, lantent_dimension=2)

src_path = r'./imaging_results/16/latent_16_Epoch_5.csv'
x, y = data_import(src_path)
cluster = KMeans(n_clusters=10, random_state=42).fit(x)
# pred for record train
clu_pred = cluster.predict(x)
plot_clustering_cm(y, clu_pred, lantent_dimension=16)

src_path = r'./imaging_results/64/latent_64_Epoch_5.csv'
x, y = data_import(src_path)
cluster = KMeans(n_clusters=10, random_state=42).fit(x)
# pred for record train
clu_pred = cluster.predict(x)
plot_clustering_cm(y, clu_pred, lantent_dimension=64)

src_path = r'./imaging_results/256/latent_256_Epoch_5.csv'
x, y = data_import(src_path)
cluster = KMeans(n_clusters=10, random_state=42).fit(x)
# pred for record train
clu_pred = cluster.predict(x)
plot_clustering_cm(y, clu_pred, lantent_dimension=256)