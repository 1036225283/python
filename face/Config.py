import model as models


DATA_SIZE = 1600
# DATA_SIZE = 304
IMAGE_SIZE = 112
BATCH_SIZE = 1
EPOCH = 400

model = models.Point68_y1
# model = models.Point68_residual1
MODEL_SAVE_PATH = model.path

print(MODEL_SAVE_PATH)
