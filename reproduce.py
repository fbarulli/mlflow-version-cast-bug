import time
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("http://mlflow:5000")

time.sleep(5)

client = mlflow.MlflowClient()

mlflow.set_experiment("test")

with mlflow.start_run():

    X, y = load_diabetes(return_X_y=True)

    model = RandomForestRegressor()
    model.fit(X, y)

    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="demo-model",
    )

versions = client.search_model_versions("name='demo-model'")

print("Versions:")
for v in versions:
    print(v.name, v.version, type(v.version))

version = max(
    versions,
    key=lambda v: int(v.version),
).version

print("Using version:", version, type(version))

print("Fetching model version...")

mv = client.get_model_version(
    "demo-model",
    version,
)

print("Fetched:", mv)

print("Setting alias...")

client.set_registered_model_alias(
    name="demo-model",
    alias="champion",
    version=version,
)

print("SUCCESS")
