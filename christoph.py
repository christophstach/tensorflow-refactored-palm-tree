from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from sklearn import decomposition
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential


def prepare_data_frame(path: str, test=False) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path)

    all_columns = [
        'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd',
        'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
        'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
        'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
    ]

    if not test:
        all_columns.append('SalePrice')

    categorical_columns = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
        'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'Functional', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'PoolQC', 'Fence', 'MiscFeature',
        'MiscVal', 'SaleType', 'SaleCondition'
    ]

    df = df[all_columns]

    for column in categorical_columns:
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.codes

    df.fillna(-1, inplace=True)
    df = df.astype(float)
    return df


def get_train_test() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train_x = prepare_data_frame('data/train.csv')
    df_train_y: pd.DataFrame = df_train_x[['SalePrice']]

    df_train_x.drop(columns=['SalePrice'], inplace=True)
    df_train_x.to_excel('train_x.xls')
    df_train_y.to_excel('train_y.xls')

    df_test_x = prepare_data_frame('data/test.csv', test=True)

    return df_train_x, df_train_y, df_test_x

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df


def explore(df_train_x: pd.DataFrame, df_train_y: pd.DataFrame):
    fig = plt.figure()

    df_train_y_sorted: pd.DataFrame = df_train_y.copy()
    df_train_y_sorted.sort_values(by=['SalePrice'], inplace=True)
    df_train_y_sorted.reset_index(drop=True, inplace=True)

    # Show Price line plot
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('SalePrice')
    df_train_y_sorted.plot(ax=ax)

    # Show distribution of price
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title('Distribution')
    sns.distplot(df_train_y_sorted['SalePrice'], ax=ax)

    # PCA 2D
    ax = fig.add_subplot(2, 2, 3)
    pca = decomposition.PCA(n_components=2)
    pca.fit(df_train_x)
    train_x = pca.transform(df_train_x)
    ax.set_title('Principal component analysis 2D')
    ax.scatter(x=train_x[:, 0], y=train_x[:, 1])


    # PCA 3D
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1, projection='3d')
    pca = decomposition.PCA(n_components=3)
    pca.fit(df_train_x)
    train_x = pca.transform(df_train_x)
    ax.set_title('Principal component analysis 3D')
    ax.scatter(xs=train_x[:, 0], ys=train_x[:, 1], zs=train_x[:, 2])

    plt.show()


def train():
    model = Sequential([
        Activation('softmax'),
    ])


df_train_x, df_train_y, df_test_x = get_train_test()
explore(df_train_x=df_train_x, df_train_y=df_train_y)
