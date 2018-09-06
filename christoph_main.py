from christoph.models import AutoEncoder3D, AutoEncoder2D, DnnRegressor
from christoph.preprocess import get_dataset
import matplotlib.pyplot as plt
import pandas as pd

x, y, scaler = get_dataset(True, remove_outliers=False)

autoEncoder3D = AutoEncoder3D()
# autoEncoder3D.fit(df_train_x.values)


autoEncoder2D = AutoEncoder2D()
# autoEncoder2D.fit(df_train_x.values)

dnnRegressor = DnnRegressor()
# dnnRegressor.fit(x, y, epochs=10000)

predictions = dnnRegressor.model.predict(x)


result = pd.DataFrame(predictions, columns=['SalePrice'])
# result.reset_index(inplace=True, drop=True)
result['Id'] = result.index + 1461
result.set_index('Id', inplace=True)

result.to_csv('./christoph/results/dnn-regressor.csv', float_format='%.9f')

print(result.head(5))

'''
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Reality vs Prediction')
ax.plot(predictions, y, 'ro')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.show()

'''

