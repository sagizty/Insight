import os
import pandas as pd
from scipy import stats


def find_cls(list_of_cls):
    cls_group = []
    for i in list_of_cls:
        if i.split('_')[0] not in cls_group:
            cls_group.append(i.split('_')[0])

    print('class names:', cls_group)
    return cls_group


def find_items_by_group(cls_group, list_of_cls):
    items_by_group = []

    for k in cls_group:
        group_cluster = []
        for i in list_of_cls:
            if i.split('_')[0] == k:
                group_cluster.append(i)
        items_by_group.append(group_cluster)

    return items_by_group


def p_val_gene_check(data_path, Norm=False, Test_strategy='compare', popmean=0.5):
    # read in:
    df = pd.read_csv(data_path)

    # wash empty and '!?'
    bef_rm = len(df)
    df.dropna(how='any', axis=0, inplace=True)
    aft_rm = len(df)
    print('num of rows removed:', bef_rm - aft_rm)
    # df.info()

    # Norm (group norm by cls)
    list_of_cls = df.columns[2:]
    # read all the classes names
    cls_group = find_cls(list_of_cls)
    # read samples group by the class
    items_by_group = find_items_by_group(cls_group, list_of_cls)

    if Norm == 'cls_gene':
        # norm by group and gene
        for group in items_by_group:  # norm by group
            for i in range(len(df[group])):  # by gene
                df.loc[i, group] = (df[group].iloc[i] - df[group].iloc[i].min()) \
                                   / (df[group].iloc[i].max() - df[group].iloc[i].min())

    elif Norm == 'cls':
        # norm by group
        for group in items_by_group:
            df[group] = (df[group] - df[group].min()) / (df[group].max() - df[group].min())

    elif Norm == 'gene':
        # norm by gene
        for i in range(len(df)):  # by gene
            df.iloc[i, 2:] = (df.iloc[i, 2:] - df.iloc[i, 2:].min()) / (df.iloc[i, 2:].max() - df.iloc[i, 2:].min())

    elif Norm == 'all':
        df.iloc[:, 2:] = (df.iloc[:, 2:] - df.iloc[:, 2:].min()) / (df.iloc[:, 2:].max() - df.iloc[:, 2:].min())

    else:
        pass

    # threshold
    df['p_value'] = 0.0

    if Test_strategy == 'compare':  # separately by CLS group then compare gene
        for idx in range(len(df)):
            tStat, pValue = stats.ttest_ind(df[items_by_group[0]].iloc[idx], df[items_by_group[1]].iloc[idx], axis=0)
            df.iloc[idx, -1] = pValue

    else:  # separately by gene
        for idx in range(len(df)):
            tStat, pValue = stats.ttest_1samp(df.iloc[idx, 2:], popmean=popmean, axis=0)
            # stats.ttest_1samp
            df.iloc[idx, -1] = pValue

    df.sort_values(by="p_value", inplace=True, ascending=True)
    # print(df.head(20))

    p_val_gene_10 = df['Probe'].iloc[:10].tolist()
    p_val_gene_50 = df['Probe'].iloc[:50].tolist()
    p_val_gene_100 = df['Probe'].iloc[:100].tolist()

    p_val_gene_10_50_100_pack = [p_val_gene_10, p_val_gene_50, p_val_gene_100]

    return p_val_gene_10_50_100_pack


def list_compare(pack_1, pack_2, name1, name2):
    compare_group = ['p_val_gene_10', 'p_val_gene_50', 'p_val_gene_100']

    for compare_group_idx in range(len(pack_1)):
        same_list = []
        conflict_list_only_in_1 = []
        conflict_list_only_in_2 = []

        for gene in pack_1[compare_group_idx]:
            if gene in pack_2[compare_group_idx]:
                same_list.append(gene)
            else:
                conflict_list_only_in_1.append(gene)

        for gene in pack_2[compare_group_idx]:
            if gene in pack_1[compare_group_idx]:
                pass
            else:
                conflict_list_only_in_2.append(gene)

        print('in' + compare_group[compare_group_idx], len(same_list), 'gene in the same group:', same_list)
        print('genes prominent only in '+name1+':', conflict_list_only_in_1)
        print('genes prominent only in '+name2+':', conflict_list_only_in_2)


def data_p_val_compare(data_path1, data_path2, Norm=False, Test_strategy='compare'):

    name1 = os.path.split(data_path1)[-1]
    name2 = os.path.split(data_path2)[-1]

    print("*********************************{}*************************************".format(
        'Norm by ' + str(Norm) + '  &   t-test by ' + Test_strategy))
    print("*********************************{}*************************************".format(
        'dataset1 of ' + name1 + '  &   dataset2 of  ' + name2))

    print('dataset of ', name1)
    pack_1 = p_val_gene_check(data_path1, Norm=Norm, Test_strategy=Test_strategy)
    print('\n')
    print('dataset of ', name2)
    pack_2 = p_val_gene_check(data_path2, Norm=Norm, Test_strategy=Test_strategy)
    print('\n')
    print('Comparing ')
    list_compare(pack_1, pack_2, name1, name2)


if __name__ == '__main__':
    # data path
    data_path1 = r'/Users/zhangtianyi/Desktop/BS6202/Assignment 3/Subtype-AllenData.csv'
    data_path2 = r'/Users/zhangtianyi/Desktop/BS6202/Assignment 3/Subtype-MaryData.csv'

    print('\n')

    for Test_strategy in ['separately', 'compare']:
        for Norm in ['None', 'all', 'gene', 'cls', 'cls_gene']:
            data_p_val_compare(data_path1, data_path2, Norm, Test_strategy)
            print('\n')
            print('\n')

    # data_path1 = r'/Users/zhangtianyi/Desktop/BS6202/Assignment 3/Leuk-ArmstrongData.csv'
    # data_path2 = r'/Users/zhangtianyi/Desktop/BS6202/Assignment 3/Leuk-GolubData.csv'

    # data_path1 = r'/Users/zhangtianyi/Desktop/BS6202/Assignment 3/Subtype-AllenData.csv'
    # data_path2 = r'/Users/zhangtianyi/Desktop/BS6202/Assignment 3/Subtype-MaryData.csv'
