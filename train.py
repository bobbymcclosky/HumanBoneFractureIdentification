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
    data='Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone Fractures Detection/data.yaml',
    epochs=100,          # longer training
    imgsz=640,
    batch=8,  
    device=0,
    name='fracture_best_model', 
    lr0=.001,           # lower LR for fine-tuning
    lrf=.0001,           # final LR (cosine schedule)
    freeze = 10,          # freeze early layers (transfer learning)
    patience = 15,        # early stopping on val metric
    dropout=.5,        
    mosaic=.5,      
    mixup=.1,      
    verbose=True
)

metrics = model.val()

print("✓ Test training and evaluation complete!")
