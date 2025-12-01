from comet_ml import Experiment
from ultralytics import YOLO

experiment = Experiment(
    api_key="UXWTiPNthHanNObk56kiYP6Kh",
    project_name="general",
    workspace="bobbymcclosky",
)

model = YOLO('yolo11n.pt')

# Train normally without a 'logger' argument
results = model.train(
    data='Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone_Fractures_Overfit_Check/data_small.yaml',
    epochs=50,          # longer training
    imgsz=640,
    batch=8,
    device=0,
    name='overfit_check',
    lr0=1e-4,           # lower LR for fine-tuning
    lrf=1e-5,           # final LR (cosine schedule)
    freeze = 0,          # freeze early layers (transfer learning)
    patience = 0,        # early stopping on val metric
    dropout=0,        # no dropout for overfitting check
    mosaic=False,      # disable mosaic for overfitting check
    mixup=False,      # disable mixup for overfitting check
    verbose=True
)

metrics = model.val()

print("✓ Test training and evaluation complete!")
