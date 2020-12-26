import model as models


DATA_SIZE = 300
IMAGE_SIZE = 224
BATCH_SIZE = 1
EPOCH = 800

model = models.Point68_residual
MODEL_SAVE_PATH = model.path

print(MODEL_SAVE_PATH)
