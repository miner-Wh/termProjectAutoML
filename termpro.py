
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import preprocessing
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import *
import warnings

warnings.filterwarnings('ignore')


# # 사용 가능 스케일러  'standard','minmax','maxabs','robust'
# # 사용 가능 인코더 'onehot', ordinal'


def myPreprocess1(dataset, num_process, cat_process):
    # 카테고리 와 넘버릭 분류
    X_cats = dataset.select_dtypes(np.object).copy()
    X_nums = dataset.select_dtypes(exclude=np.object).copy()

    if num_process == 'standard':
        scaler = preprocessing.StandardScaler()
    elif num_process == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    elif num_process == 'maxabs':
        scaler = preprocessing.MaxAbsScaler()
    elif num_process == 'robust':
        scaler = preprocessing.RobustScaler()
    else:
        raise ValueError("Supported 'num_process' : 'standard','minmax','maxabs','robust'")
    if cat_process == 'onehot':
        encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    elif cat_process == 'ordinal':
        encoder = preprocessing.OrdinalEncoder()
    else:
        raise ValueError("Supported 'cat_process' : 'onehot', ordinal'")

    X_nums = scaler.fit_transform(X_nums)
    X_cats = encoder.fit_transform(X_cats)

    train_processed = np.concatenate([X_nums, X_cats], 1)

    return train_processed


def eval_classification(name, model, pred, ytest):
    print('[' + name + ']')
    print("Accuracy (Test Set): %.4f" % accuracy_score(ytest, pred))
    print("Precision (Test Set): %.4f" % precision_score(ytest, pred))
    print("Recall (Test Set): %.4f" % recall_score(ytest, pred))
    print("F1-Score (Test Set): %.4f" % f1_score(ytest, pred))

    fpr, tpr, thresholds = roc_curve(ytest, pred, pos_label=1)  # pos_label: label yang kita anggap positive
    print("AUC: %.2f" % auc(fpr, tpr))


def show_best_hyperparameter(model, hyperparameters):
    for key, value in hyperparameters.items():
        print('Best ' + key + ':', model.get_params()[key])


def show_cmatrix(ytest, pred):
    # Creating confusion matrix
    cm = confusion_matrix(ytest, pred)

    # Putting the matrix a dataframe form
    cm_df = pd.DataFrame(cm, index=['Actually Not Canceled', 'Actually Canceled'],
                         columns=['Predicted Not Canceled', 'Predicted Canceled'])

    # visualizing the confusion matrix
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10, 6))

    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", xticklabels=cm_df.columns, yticklabels=cm_df.index,
                annot_kws={"size": 20})
    plt.title("Confusion Matrix", size=20)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()


def clustering_plot(df, model_name, model, param, labels, cluster, score):
    print('[', model_name, ']')

    if model_name == 'K_Means':
        cluster_id = pd.DataFrame(cluster.labels_)
        d1 = pd.concat([df, cluster_id], axis=1)
        d1.columns = [0, 1, "cluster"]
        sns.scatterplot(d1[0], d1[1], hue=d1['cluster'], legend="full")
        sns.scatterplot(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], label='Centroids')
        plt.title("KMeans Clustering with k = {}".format(param))
        plt.legend()
        plt.show()

        print('K = ', param)

    elif model_name == 'GMM':
        frame = pd.DataFrame(df)
        frame['cluster'] = labels
        frame.columns = [df.columns[0], df.columns[1], 'cluster']

        plt.title('GMM with K = {}'.format(param))
        for i in range(0, param + 1):
            data = frame[frame["cluster"] == i]
            plt.scatter(data[data.columns[0]], data[data.columns[1]])
        plt.show()
        print("K = ", param)

    elif model_name == 'DBSCAN':
        cluster_id = pd.DataFrame(cluster.labels_)

        d2 = pd.DataFrame()
        d2 = pd.concat([df, cluster_id], axis=1)
        d2.columns = [0, 1, "cluster"]
        sns.scatterplot(d2[0], d2[1], hue=d2['cluster'], legend="full")
        plt.title('DBSCAN with eps = {}, min_samples = {}'.format(param['eps'], param['min_samples']))
        plt.show()

        labels = d2['cluster']
        print('eps = {}, min_samples = {}'.format(param['eps'], param['min_samples']))


    elif model_name == 'MeanShift':
        cluster_id = pd.DataFrame(cluster.labels_)

        d6 = pd.DataFrame()
        d6 = pd.concat([df, cluster_id], axis=1)
        d6.columns = [0, 1, "cluster"]

        sns.scatterplot(d6[0], d6[1], hue=d6['cluster'], legend="full")
        plt.title('Mean Shift with {} samples'.format(param))
        plt.show()

        print('n_samples(estimate_bandwidth) = {}'.format(param))

    print('Silhouette Score(euclidean):', metrics.silhouette_score(df, labels, metric='euclidean'))
    print('Silhouette Score(manhattan):', metrics.silhouette_score(df, labels, metric='manhattan'))
    print('Purity_score:', score)


def show_RocCurve(model, pred_X_test, Y_test):
    plot = plot_roc_curve(model, pred_X_test, Y_test)
    plt.show()


def initial_tuning(model, hyperparameters, X_train_val, Y_train_val):
    # Find initial hyperparams using GridSearchCV
    initial_tuned_model = GridSearchCV(model, param_grid=hyperparameters, cv=3)
    initial_tuned_model.fit(X_train_val, Y_train_val)

    result = pd.DataFrame(initial_tuned_model.cv_results_['params'])
    result['mean_test_score'] = initial_tuned_model.cv_results_['mean_test_score']
    result = result.sort_values(by='mean_test_score', ascending=False)

    return result


def initial_tuning_2(df, y, model_name, hyperparams):
    # Apply various hyperparameters in each models
    print('-------------------------------')
    print('Start Initial Tuning')
    print('-------------------------------')
    _1st_param = 0
    _2nd_param = 0
    _1st_score = 0
    _2nd_score = 0
    init_best_model = ''

    if model_name == 'K_Means':
        distortions = []
        for k in hyperparams['k']:
            kmeans = KMeans(n_clusters=k, init='k-means++')
            cluster = kmeans.fit(df)
            labels = kmeans.predict(df)
            score = purity_score(y, labels)
            print('(K=', k, ') Purity_score:', score)
            if score >= _1st_score:
                _2nd_param = _1st_param
                _2nd_score = _1st_score
                _1st_score = score
                _1st_param = k
                init_best_model = kmeans

            elif score >= _2nd_score:
                _2nd_score = score
                _2nd_param = k


    elif model_name == 'GMM':
        for k in hyperparams['k']:
            gmm = GaussianMixture(n_components=k)
            gmm.fit(df)
            labels = gmm.predict(df)
            score = purity_score(y, labels)
            print('(K=', k, ') Purity_score:', score)
            if score >= _1st_score:
                _2nd_param = _1st_param
                _2nd_score = _1st_score
                _1st_score = score
                _1st_param = k
                init_best_model = gmm

            elif score >= _2nd_score:
                _2nd_score = score
                _2nd_param = k


    elif model_name == 'DBSCAN':
        eps = hyperparams['DBSCAN_params']['eps']
        minsam = hyperparams['DBSCAN_params']['min_samples']

        for i in eps:
            for j in minsam:
                db = DBSCAN(eps=i, min_samples=j)
                cluster = db.fit(df)
                cluster_id = pd.DataFrame(cluster.labels_)

                d2 = pd.DataFrame()
                d2 = pd.concat([df, cluster_id], axis=1)
                d2.columns = [0, 1, "cluster"]
                score = purity_score(y, d2['cluster'])
                print('(eps = ', i, ' min_samples =', j, ') Purity_score:', score)

                if score >= _1st_score:
                    _2nd_param = _1st_param
                    _2nd_score = _1st_score
                    _1st_score = score
                    _1st_param = {'eps': i, 'min_samples': j}
                    init_best_model = db

                elif score >= _2nd_score:
                    _2nd_score = score
                    _2nd_param = {'eps': i, 'min_samples': j}


    elif model_name == 'MeanShift':
        n = hyperparams['MeanShift_params']['n']
        for i in n:
            bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=i)
            ms = MeanShift(bandwidth=bandwidth)
            cluster = ms.fit(df)
            cluster_id = pd.DataFrame(cluster.labels_)

            d6 = pd.DataFrame()
            d6 = pd.concat([df, cluster_id], axis=1)
            d6.columns = [0, 1, "cluster"]
            score = purity_score(y, d6['cluster'])

            print('(n_samples =', i, ') Purity_score:', score)

            if score >= _1st_score:
                _2nd_param = _1st_param
                _2nd_score = _1st_score
                _1st_score = score
                _1st_param = i
                init_best_model = ms

            elif score >= _2nd_score:
                _2nd_score = score
                _2nd_param = i

    return _1st_param, _1st_score, _2nd_param, _2nd_score, init_best_model


def tuning(initial_result, model_name, model, X_train_val, Y_train_val):
    _1st_param = initial_result.iloc[0, 0]
    _2nd_param = initial_result.iloc[1, 0]
    _1st_score = initial_result.iloc[0, 1]
    _2nd_score = initial_result.iloc[1, 1]

    # Split to validation dataset and train
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, shuffle=True,
                                                      random_state=100)

    best_score = 0
    best_param = -1
    second_score = 0
    second_param = -1

    number = 20
    if model_name == 'Decision Tree':
        number = 4
    eps = 0.0003
    k = 0
    best_model = model

    while True:
        k = k + 1
        if _1st_param - _2nd_param < 0:
            if model_name == 'Decision Tree':
                hyperparameters = np.arange(_1st_param, _2nd_param + 1, number)
            else:
                hyperparameters = np.linspace(_1st_param, _2nd_param, num=number)

        else:
            if model_name == 'Decision Tree':
                hyperparameters = np.arange(_2nd_param, _1st_param + 1, number)
            else:
                hyperparameters = np.linspace(_2nd_param, _1st_param, num=number)

        # indexes = [0, -1]
        # hyperparameters = np.delete(hyperparameters, indexes)

        for i in range(len(hyperparameters)):
            param = hyperparameters[i]
            if param - best_param == 0 or param - second_param == 0:
                continue
            if model_name == 'Decision Tree':
                model.max_depth = param
            else:
                model.C = param

            model = model.fit(X_train, y_train)
            # print(model)

            score = model.score(X_val, y_val)

            # print(score)
            if score >= best_score:
                second_score = best_score
                second_param = best_param
                best_score = score
                best_param = param
                best_model = model

            elif score >= second_score:
                second_score = score
                second_param = param

        print("Iteration ", k)
        print('hyperparameters:', hyperparameters)
        print('\nbest_score:', best_score)
        print('best_param:', best_param)
        print('second_score:', second_score)
        print('second_param:', second_param)
        print('---------------------------------------')

        if k != 1 and (best_score - _1st_score) / _1st_score <= eps:
            break

        _1st_param = best_param
        _1st_score = best_score
        _2nd_param = second_param
        _2nd_score = second_score
        if model_name == 'Decision Tree':
            number = number / 2

    return best_score, best_model


def tuning2(df, y, model, model_name, _1st_score, _1st_param, _2nd_score, _2nd_param):
    best_score = 0
    best_param = -1
    second_score = 0
    second_param = -1
    best_labels = 0
    best_model = model
    best_cluster = ''
    hyperparameters = ''

    number = 20
    interval = 4

    train_eps = 0.0001
    iter = 0
    print('-------------------------------')
    print('Start Tuning ')
    print('-------------------------------')
    # print('model_name : ',model_name)
    print()

    while True:
        iter = iter + 1

        if model_name == 'DBSCAN':

            if _1st_param['min_samples'] - _2nd_param['min_samples'] < 0:
                hyperparameters_min_samples = np.arange(_1st_param['min_samples'], _2nd_param['min_samples'] + 1,
                                                        interval)
                hyperparameters_min_samples = list(map(int, hyperparameters_min_samples))
            else:
                hyperparameters_min_samples = np.arange(_2nd_param['min_samples'], _1st_param['min_samples'] + 1,
                                                        interval)
                hyperparameters_min_samples = list(map(int, hyperparameters_min_samples))

            if _1st_param['eps'] - _2nd_param['eps'] < 0:
                hyperparameters_eps = np.linspace(_2nd_param['eps'], _1st_param['eps'], num=number)
            else:
                hyperparameters_eps = np.linspace(_2nd_param['eps'], _1st_param['eps'], num=number)

            hyperparameters = dict(eps=hyperparameters_eps, min_samples=hyperparameters_min_samples)

            for i in range(len(hyperparameters_min_samples)):
                min_samples = hyperparameters_min_samples[i]
                model.min_samples = min_samples

                for j in range(len(hyperparameters_eps)):
                    eps = hyperparameters_eps[j]
                    model.eps = eps

                    param = {'eps': eps, 'min_samples': min_samples}

                    if best_param != -1:
                        if param['eps'] - best_param['eps'] == 0 and param['min_samples'] == best_param['min_samples']:
                            continue
                    if second_param != -1:
                        if param['eps'] == second_param['eps'] and param['min_samples'] == second_param['min_samples']:
                            continue

                    cluster = model.fit(df)
                    cluster_id = pd.DataFrame(cluster.labels_)
                    d2 = pd.DataFrame()
                    d2 = pd.concat([df, cluster_id], axis=1)
                    d2.columns = [0, 1, "cluster"]

                    score = purity_score(y, d2['cluster'])

                    if score >= best_score:
                        second_score = best_score
                        second_param = best_param
                        best_score = score
                        best_param = param
                        best_model = model
                        best_cluster = cluster


                    elif score >= second_score:
                        second_score = score
                        second_param = param



        else:

            if _1st_param - _2nd_param < 0:
                hyperparameters = np.arange(_1st_param, _2nd_param + 1, interval)
            else:
                hyperparameters = np.arange(_2nd_param, _1st_param + 1, interval)

            hyperparameters = list(map(int, hyperparameters))
            print("Iteration ", iter)
            print('hyperparameters:', hyperparameters)

            for i in range(len(hyperparameters)):
                param = hyperparameters[i]
                if param - best_param == 0 or param - second_param == 0:
                    continue
                if model_name == 'K_Means':
                    model.n_cluster = param
                elif model_name == 'GMM':
                    model.n_components = param
                elif model_name == 'MeanShift':
                    bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=param)
                    model.bandwidth = bandwidth
                else:
                    model.C = param

                cluster = model.fit(df)
                labels = model.predict(df)
                score = purity_score(y, labels)

                if score >= best_score:
                    second_score = best_score
                    second_param = best_param
                    best_score = score
                    best_param = param
                    best_model = model
                    best_labels = labels
                    best_cluster = cluster


                elif score >= second_score:
                    second_score = score
                    second_param = param

        print('\nbest_score:', best_score)
        print('best_param:', best_param)
        print('second_score:', second_score)
        print('second_param:', second_param)
        print('---------------------------------------')

        if iter != 1 and (best_score - _1st_score) / _1st_score <= train_eps:
            break

        _1st_param = best_param
        _1st_score = best_score
        _2nd_param = second_param
        _2nd_score = second_score
        interval = interval / 2

    return best_score, best_param, best_model, best_labels, best_cluster


def autoML(models, supervised, hyperparams,dataset,Y):
    hyperparameters = ''
    model = ''
    model_name = ''

    if supervised == 'classification':
        for i in range(len(models)):
            X_train_val, X_test, Y_train_val, Y_test = train_test_split(dataset, Y, test_size=0.3, shuffle=True,
                                                                        stratify=Y,
                                                                        random_state=34)
            model_name = models[i]
            if model_name == 'Logistic Regression':
                # c
                hyperparameters = hyperparams['LR_params']
                model = LogisticRegression(solver='liblinear', random_state=100)


            elif model_name == 'Decision Tree':
                # max_Depth
                hyperparameters = hyperparams['DT_params']
                model = DecisionTreeClassifier(random_state=100)

            elif model_name == 'SVM':
                # c
                hyperparameters = hyperparams['SVM_params']
                model = SVC(kernel='linear', random_state=100)

            result = initial_tuning(model, hyperparameters, X_train_val, Y_train_val)
            print(result)

            best_score, best_model = tuning(result, model_name, model, X_train_val, Y_train_val)

            y_pred = best_model.predict(X_test)

            eval_classification(model_name, model, y_pred, Y_test)

            print('Train score: ' + str(best_model.score(X_train_val, Y_train_val)))  # accuracy
            print('Test score:' + str(best_model.score(X_test, Y_test)))  # accuracy

            show_cmatrix(Y_test, y_pred)
            show_RocCurve(best_model, X_test, Y_test)


    elif supervised == 'clustering':
        # PCA
        pca = PCA(n_components=2)
        reduced_df = pca.fit_transform(dataset)
        df = pd.DataFrame(reduced_df)
        y=Y

        for i in range(len(models)):
            model_name = models[i]
            print('\nModel name:', model_name)

            _1st_param, _1st_score, _2nd_param, _2nd_score, init_best_model = initial_tuning_2(df, y, model_name,
                                                                                               hyperparams)

            print('Initial Best Hyperparameters:', _1st_param, ' Score:', _1st_score)
            print('Initial Second Hyperparameters:', _2nd_param, ' Score:', _2nd_score)

            best_score, best_param, best_model, best_labels, best_cluster = tuning2(df, y, init_best_model, model_name,
                                                                                    _1st_score, _1st_param, _2nd_score,
                                                                                    _2nd_param)

            print('---------------------------------------')
            print('\nBest Result')
            print('---------------------------------------')
            clustering_plot(df, model_name, best_model, best_param, best_labels, best_cluster, best_score)


def elbow_curve(distortions):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 5), distortions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


################
# 데이터셑 불러 오기
df_original = pd.read_csv('/content/drive/MyDrive/hotel_booking.csv', encoding='utf-8')

# 결측치 확인
# print("Check dirty data : \n" + str(df_original.isna().sum()))

# 오리지널 카피 for 클린 데이터
df_clean = df_original.copy()

################################ clean
# children 평균 값으로 채우기
df_clean['children'] = df_clean['children'].fillna(df_clean['children'].mean())
# agent 0 값으로 채우기
df_clean['agent'] = df_clean['agent'].fillna(0)
# company 0 값으로 채우기
df_clean['company'] = df_clean['company'].fillna(0)

# country 결측 행 제거
df_clean = df_clean.dropna(axis=0)

# 결측치 확인
# print("Check dirty data : \n" + str(df_clean.isna().sum()))
################################

################################ data type setting
# print(df_clean.dtypes)

# 카테고리컬 인데 넘버리컬로 된거 변환
df_clean = df_clean.astype({'agent': 'object', 'company': 'object', 'is_repeated_guest': 'object'})

# print(df_clean.dtypes)
################################


# # # ######SVC sampling###############
df_clean = df_clean.sample(frac=0.5, random_state=1)
# #
# print(df_clean.shape)


################################ use preprocess()

# split First
# X에 들어갈 attribute는 사전에 준비한 그룹 별로 선택하여 사용
# 고객1
X_c1 = df_clean[['adults', 'children', 'babies', 'country', 'customer_type']]
# 고객2
X_c2 = df_clean[
    ['stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'country', 'customer_type',
     'previous_cancellations', 'previous_bookings_not_canceled']]
# 시기
X_w = df_clean[['arrival_date_month', 'stays_in_weekend_nights', 'stays_in_week_nights']]
# 서비스
X_s = df_clean[
    ['meal', 'reserved_room_type', 'assigned_room_type', 'required_car_parking_spaces', 'total_of_special_requests',
     'adr']]
# 예약정보
X_r = df_clean[
    ['meal', 'reserved_room_type', 'assigned_room_type', 'required_car_parking_spaces', 'total_of_special_requests',
     'market_segment', 'distribution_channel', 'booking_changes', 'deposit_type', 'agent', 'company', 'adr',
     'reservation_status']]

Y = df_clean['is_canceled']

new_dataset = myPreprocess1(X_c1, 'standard', 'ordinal')


classification_models = ['Logistic Regression', 'Decision Tree', 'SVM']


classification_hyperparams = {

    'LR_params': dict(C=[0.01, 0.1, 1, 10, 100]),
    'DT_params': dict(max_depth=[2, 10, 100]),
    'SVM_params': dict(C=[0.01, 0.1, 1, 10, 100])

}

clustering_models = ['K_Means', 'DBSCAN', 'MeanShift', 'GMM']

clustering_hyperparams = {

    'DBSCAN_params': {
        'eps': [0.0001, 0.001, 0.01, 0.1, 1, 10],
        'min_samples': [10, 50, 100]
        # 'eps':[0.1, 0.2, 0.3, 0.4, 0.5]
    },
    'MeanShift_params': {
        'n': [20, 50, 100]
    },
    'k': [2, 10, 100, 500]
}

autoML(classification_models, 'classification', classification_hyperparams,new_dataset,Y)
autoML(clustering_models, 'clustering', clustering_hyperparams,new_dataset,Y)

