



# 새로운 모델로 교체 (make_model_v2.py 생성)
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

X = np.random.randn(200, 3)
y = (X[:,0] + X[:,1] > 0).astype(int)

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("v2 모델 생성 완료!")


