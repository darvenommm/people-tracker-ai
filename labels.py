from torchvision.models.video import r3d_18, R3D_18_Weights

# Загрузим веса
weights = R3D_18_Weights.DEFAULT

# Получим список всех 400 категорий
labels = weights.meta["categories"]

for label in labels:
    print(label)
