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
# from pyclustering.cluster.clarans import clarans;
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')

# ################
# # 사용 가능 스케일러  'standard','minmax','maxabs','robust'
# # 사용 가능 인코더 'onehot', ordinal'
# def myPreprocess(train, test, num_process, cat_process):
#     # 카테고리 와 넘버릭 분류
#     X_cats = train.select_dtypes(np.object).copy()
#     X_nums = train.select_dtypes(exclude=np.object).copy()
#
#     X_cats_t = test.select_dtypes(np.object).copy()
#     X_nums_t = test.select_dtypes(exclude=np.object).copy()
#
#     if num_process == 'standard':
#         scaler = preprocessing.StandardScaler()
#     elif num_process == 'minmax':
#         scaler = preprocessing.MinMaxScaler()
#     elif num_process == 'maxabs':
#         scaler = preprocessing.MaxAbsScaler()
#     elif num_process == 'robust':
#         scaler = preprocessing.RobustScaler()
#     else:
#         raise ValueError("Supported 'num_process' : 'standard','minmax','maxabs','robust'")
#     if cat_process == 'onehot':
#         encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
#     elif cat_process == 'ordinal':
#         encoder = preprocessing.OrdinalEncoder()
#     else:
#         raise ValueError("Supported 'cat_process' : 'onehot', ordinal'")
#
#     X_nums = scaler.fit_transform(X_nums)
#     X_cats = encoder.fit_transform(X_cats)
#
#     X_nums_t = scaler.transform(X_nums_t)
#     X_cats_t = encoder.transform(X_cats_t)
#
#     train_processed = np.concatenate([X_nums, X_cats], 1)
#     test_processed = np.concatenate([X_nums_t, X_cats_t], 1)
#
#     return train_processed, test_processed


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


def tuning(initial_result,model_name, model, X_train_val, Y_train_val):
    _1st_param = initial_result.iloc[0, 0]
    _2nd_param = initial_result.iloc[1, 0]
    _1st_score = initial_result.iloc[0, 1]
    _2nd_score = initial_result.iloc[1, 1]
    # _1st_param = 10
    # _2nd_param = 100
    # _1st_score = 0
    # _2nd_score = 0

    # Split to validation dataset and train
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25,shuffle=True,random_state=100)

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
            if param-best_param==0 or param-second_param==0:
                continue
            if model_name == 'Decision Tree':
                model.max_depth = param
            else:
                model.C = param

            model = model.fit(X_train, y_train)
            #print(model)

            score = model.score(X_val, y_val)

            #print(score)
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

        if k!=1 and (best_score - _1st_score) / _1st_score <= eps:
            break

        _1st_param = best_param
        _1st_score = best_score
        _2nd_param = second_param
        _2nd_score = second_score
        if model_name == 'Decision Tree':
            number = number / 2

    return best_score, best_model


def autoML(models):
    hyperparameters = ''
    model = ''
    model_name = ''
    for i in range(len(models)):
        model_name = models[i]
        if model_name == 'Logistic Regression':
            # c
            hyperparameters = dict(C=[0.01, 0.1, 1, 10, 100])
            model = LogisticRegression(solver='liblinear', random_state=100)


        elif model_name == 'Decision Tree':
            # max_Depth
            hyperparameters = dict(max_depth=[2, 10, 100])
            model = DecisionTreeClassifier(random_state=100)

        elif model_name == 'SVM':
            # c
            hyperparameters = dict(C=[0.01, 0.1, 1, 10, 100])
            model = SVC(kernel='linear', random_state=100)

        result = initial_tuning(model, hyperparameters, X_train_val, Y_train_val)
        print(result)

        best_score, best_model = tuning(result,model_name, model, X_train_val, Y_train_val)

        y_pred = best_model.predict(X_test)


        eval_classification(model_name, model, y_pred, Y_test)

        print('Train score: ' + str(best_model.score(X_train_val, Y_train_val)))  # accuracy
        print('Test score:' + str(best_model.score(X_test, Y_test)))  # accuracy

        show_cmatrix(Y_test, y_pred)

        show_RocCurve(best_model, X_test, Y_test)





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


def clustering(df,y, models, hyperparams):

    # Experiment with various models
    for model in models:
        print("Current model: ", model)
        # Apply various hyperparameters in each models
        if model == 'K_Means':
            distortions = []
            for k in hyperparams['k']:
                kmeans = KMeans(n_clusters=k, init='k-means++')
                cluster = kmeans.fit(df)
                labels = kmeans.predict(df)
                cluster_id = pd.DataFrame(cluster.labels_)
                distortions.append(kmeans.inertia_)

                d1 = pd.concat([df, cluster_id], axis=1)
                #print(d1)

                d1.columns = [0, 1, "cluster"]

                sns.scatterplot(d1[0], d1[1], hue=d1['cluster'], legend="full")
                sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], label='Centroids')
                plt.title("KMeans Clustering with k = {}".format(k))
                plt.legend()
                plt.show()

                print('Silhouette Score(euclidean):', metrics.silhouette_score(df, labels, metric='euclidean'), " (", k,
                      "-clusters)")
                print('Silhouette Score(manhattan):', metrics.silhouette_score(df, labels, metric='manhattan'))

                print('Quantile comparison score(purity_score):', purity_score(y, labels))

            elbow_curve(distortions)



        elif model == 'GMM':
            for k in hyperparams['k']:
                gmm = GaussianMixture(n_components=k)
                gmm.fit(df)
                labels = gmm.predict(df)

                frame = pd.DataFrame(df)
                frame['cluster'] = labels
                frame.columns = [df.columns[0], df.columns[1], 'cluster']

                plt.title('GMM with K = {}'.format(k))
                for i in range(0, k + 1):
                    data = frame[frame["cluster"] == i]
                    plt.scatter(data[data.columns[0]], data[data.columns[1]])
                plt.show()

                print('Silhouette Score(euclidean):', metrics.silhouette_score(df, labels, metric='euclidean'), " (", k,
                      "-components)")
                print('Silhouette Score(manhattan):', metrics.silhouette_score(df, labels, metric='manhattan'))

                print('Quantile comparison score(purity_score):', purity_score(y, labels))


        elif model == 'DBSCAN':
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

                    sns.scatterplot(d2[0], d2[1], hue=d2['cluster'], legend="full")
                    plt.title('DBSCAN with eps = {}, min_samples = {}'.format(i,j))
                    plt.show()


                    print('Silhouette Score(euclidean):',
                          metrics.silhouette_score(d2.iloc[:, :-1], d2['cluster'], metric='euclidean'), " (eps=", i,
                          ")", " (min_samples=", j, ")")
                    print('Silhouette Score(manhattan):',
                          metrics.silhouette_score(d2.iloc[:, :-1], d2['cluster'], metric='manhattan'))

                    print('Quantile comparison score(purity_score):', purity_score(y, d2['cluster']))



        elif model == 'MeanShift':
            n = hyperparams['MeanShift_params']['n']
            for i in n:
                bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=i)
                ms = MeanShift(bandwidth=bandwidth)
                cluster = ms.fit(df)
                cluster_id = pd.DataFrame(cluster.labels_)

                d6 = pd.DataFrame()
                d6 = pd.concat([df, cluster_id], axis=1)
                d6.columns = [0, 1, "cluster"]

                sns.scatterplot(d6[0], d6[1], hue=d6['cluster'], legend="full")
                plt.title('Mean Shift with {} samples'.format(i))
                plt.show()

                print('n_samples(estimate_bandwidth) = {}'.format(i))

                print('Silhouette Coefficient(euclidean): ',
                      metrics.silhouette_score(d6.iloc[:, :-1], d6['cluster'], metric='euclidean'))
                print('Silhouette Coefficient(manhattan): ',
                      metrics.silhouette_score(d6.iloc[:, :-1], d6['cluster'], metric='manhattan'))

                print('Quantile comparison score(purity_score):', purity_score(y, d6['cluster']))






################
# 데이터셑 불러 오기
df_original = pd.read_csv('C:/Users/Howoon/Downloads/archive (18)/hotel_booking.csv', encoding='utf-8')


# 결측치 확인
print("Check dirty data : \n" + str(df_original.isna().sum()))

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
print("Check dirty data : \n" + str(df_clean.isna().sum()))
################################

################################ data type setting
print(df_clean.dtypes)

# 카테고리컬 인데 넘버리컬로 된거 변환
df_clean = df_clean.astype({'agent': 'object', 'company': 'object', 'is_repeated_guest': 'object'})

print(df_clean.dtypes)
################################


# # # ######SVC sampling###############
df_clean=df_clean.sample(frac=0.5,random_state=1)
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



new_dataset=myPreprocess1(X_c1,'standard', 'ordinal')
X_train_val, X_test, Y_train_val, Y_test = train_test_split(new_dataset, Y, test_size=0.3, shuffle=True, stratify=Y, random_state=34)


classification_models = ['Logistic Regression','Decision Tree','SVM']
clustering_models = [ 'K_Means','DBSCAN','MeanShift','GMM']


autoML(classification_models)



clustering_hyperparams = {

    'DBSCAN_params': {
        'eps': [1,2],
        'min_samples': [5,10,50]
        # 'eps':[0.1, 0.2, 0.3, 0.4, 0.5]
    },
    'MeanShift_params': {
        'n': [20,25,15]
    },
    'k': range(2, 5)
}

y=Y
pca = PCA(n_components=2)
reduced_df = pca.fit_transform(new_dataset)
reduced_df = pd.DataFrame(reduced_df)
clustering(reduced_df,y, clustering_models, clustering_hyperparams)


