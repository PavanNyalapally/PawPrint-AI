import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras import mixed_precision

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import os
import json
import random
import logging
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

# =====================================================
# LOGGING SETUP
# =====================================================
def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure logging for training pipeline."""
    logger = logging.getLogger('PawPrintAI')
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

# =====================================================
# LOAD CONFIG
# =====================================================
def load_config(config_path: str = "config.json") -> dict:
    """Load training configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in config file: {e}")
        raise

config = load_config()

# =====================================================
# SET SEEDS FOR REPRODUCIBILITY
# =====================================================
# =====================================================
# SET SEEDS FOR REPRODUCIBILITY
# =====================================================
SEED = config['training']['seed']
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
logger.info(f"✅ Random seeds set for reproducibility (SEED={SEED})")

# =====================================================
# PERFORMANCE SETTINGS (Apple Silicon)
# =====================================================
if config['training']['mixed_precision']:
    mixed_precision.set_global_policy("mixed_float16")
    logger.info("✅ Mixed precision enabled")

# =====================================================
# EXTRACT CONFIG VALUES
# =====================================================
TRAIN_DIR = config['data']['train_dir']
VAL_DIR = config['data']['val_dir']
TEST_DIR = config['data']['test_dir']
MODEL_DIR = config['data']['model_dir']
IMAGE_SIZE = tuple(config['training']['image_size'])
BATCH_SIZE = config['training']['batch_size']
EPOCHS = config['training']['epochs']
FINE_TUNE_EPOCHS = config['training']['fine_tune_epochs']
LEARNING_RATE = config['training']['learning_rate']
LABEL_SMOOTHING = config['training']['label_smoothing']
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

os.makedirs(MODEL_DIR, exist_ok=True)
logger.info(f"✅ Model directory ready: {MODEL_DIR}")

# =====================================================
# DATA VALIDATION
# =====================================================
def validate_dataset_directory(directory: str, min_images: int = 10) -> bool:
    """Validate dataset directory structure and image count."""
    try:
        if not os.path.isdir(directory):
            logger.error(f"❌ Directory not found: {directory}")
            return False
        
        class_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if not class_dirs:
            logger.error(f"❌ No class subdirectories found in {directory}")
            return False
        
        total_images = 0
        supported_formats = tuple(config['validation']['supported_formats'])
        
        for class_dir in class_dirs:
            class_path = os.path.join(directory, class_dir)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(supported_formats)]
            
            if len(images) < min_images:
                logger.warning(f"⚠️  Class '{class_dir}' has only {len(images)} images (minimum: {min_images})")
            
            total_images += len(images)
        
        logger.info(f"✅ {directory}: {len(class_dirs)} classes, {total_images} total images")
        return True
    except Exception as e:
        logger.error(f"❌ Error validating {directory}: {e}")
        return False

def validate_all_datasets() -> bool:
    """Validate all dataset directories."""
    logger.info("\n📊 Validating datasets...")
    min_images = config['validation']['min_images_per_class']
    
    train_valid = validate_dataset_directory(TRAIN_DIR, min_images)
    val_valid = validate_dataset_directory(VAL_DIR, min_images)
    test_valid = validate_dataset_directory(TEST_DIR, min_images)
    
    if not all([train_valid, val_valid, test_valid]):
        logger.error("❌ Dataset validation failed!")
        return False
    
    logger.info("✅ All datasets validated successfully\n")
    return True

if not validate_all_datasets():
    logger.error("❌ Training cannot proceed with invalid datasets")
    sys.exit(1)

# =====================================================
# CUSTOM CALLBACK FOR DETAILED LOGGING
# =====================================================
class TrainingLogger(Callback):
    """Custom callback for detailed training logging."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.debug(
            f"Epoch {epoch + 1}: loss={logs.get('loss', 'N/A'):.4f}, "
            f"accuracy={logs.get('accuracy', 'N/A'):.4f}, "
            f"val_loss={logs.get('val_loss', 'N/A'):.4f}, "
            f"val_accuracy={logs.get('val_accuracy', 'N/A'):.4f}"
        )

# =====================================================
# DATA GENERATORS (Enhanced Augmentation)
# =====================================================
try:
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=config['augmentation']['rotation_range'],
        width_shift_range=config['augmentation']['width_shift_range'],
        height_shift_range=config['augmentation']['height_shift_range'],
        zoom_range=config['augmentation']['zoom_range'],
        horizontal_flip=config['augmentation']['horizontal_flip'],
        vertical_flip=config['augmentation']['vertical_flip'],
        shear_range=config['augmentation']['shear_range'],
        fill_mode=config['augmentation']['fill_mode']
    )

    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    logger.info("✅ Data generators created")

    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_data = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    NUM_CLASSES = train_data.num_classes
    logger.info(f"✅ Data loaded. Classes ({NUM_CLASSES}): {train_data.class_indices}")
    
except Exception as e:
    logger.error(f"❌ Error loading data: {e}")
    sys.exit(1)

# =====================================================
# SAVE CLASS INDICES (CRITICAL)
# =====================================================
try:
    class_indices_path = os.path.join(MODEL_DIR, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(train_data.class_indices, f, indent=2)
    logger.info(f"✅ Saved class indices to {class_indices_path}")
except Exception as e:
    logger.error(f"❌ Error saving class indices: {e}")
    sys.exit(1)

# =====================================================
# CLASS WEIGHTS
# =====================================================
try:
    classes = train_data.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes
    )
    class_weights = dict(enumerate(class_weights))
    logger.info(f"✅ Class weights computed: {class_weights}")
except Exception as e:
    logger.error(f"❌ Error computing class weights: {e}")
    sys.exit(1)

# =====================================================
# MODEL – EfficientNetV2-S
# =====================================================
try:
    logger.info("\n🔨 Building EfficientNetV2-S model...")
    base_model = EfficientNetV2S(
        include_top=config['model']['include_top'],
        weights=config['model']['weights'],
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(config['model']['dense_units'], activation="relu")(x)
    x = Dropout(config['model']['dropout_rate'])(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32"  # REQUIRED for mixed precision
    )(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    logger.info("✅ Model architecture created successfully")
    
except Exception as e:
    logger.error(f"❌ Error building model: {e}")
    sys.exit(1)

# =====================================================
# COMPILE – PHASE 1
# =====================================================
try:
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    logger.info("✅ Model compiled (Phase 1)")
except Exception as e:
    logger.error(f"❌ Error compiling model: {e}")
    sys.exit(1)

# =====================================================
# CALLBACKS
# =====================================================
try:
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f'best_model_{TIMESTAMP}.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config['callbacks']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config['callbacks']['reduce_lr_factor'],
            patience=config['callbacks']['reduce_lr_patience'],
            min_lr=config['callbacks']['min_lr'],
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(MODEL_DIR, f'logs_{TIMESTAMP}'),
            histogram_freq=1
        ),
        TrainingLogger()
    ]
    logger.info("✅ Callbacks configured")
except Exception as e:
    logger.error(f"❌ Error configuring callbacks: {e}")
    sys.exit(1)

# =====================================================
# TRAIN – PHASE 1
# =====================================================
try:
    logger.info("\n🔹 Phase 1: Training classifier head\n")
    history_phase1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    logger.info("✅ Phase 1 training completed")
except Exception as e:
    logger.error(f"❌ Error during Phase 1 training: {e}")
    sys.exit(1)

# =====================================================
# FINE-TUNING – PHASE 2
# =====================================================
try:
    logger.info("\n🔹 Phase 2: Fine-tuning EfficientNetV2-S\n")

    base_model.trainable = True
    fine_tune_layers = config['callbacks']['fine_tune_layers']
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / config['training']['fine_tune_learning_rate_factor']),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    logger.info(f"✅ Base model fine-tuning enabled (last {fine_tune_layers} layers)")

    history_phase2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    logger.info("✅ Phase 2 fine-tuning completed")
except Exception as e:
    logger.error(f"❌ Error during Phase 2 fine-tuning: {e}")
    sys.exit(1)

# =====================================================
# EVALUATION
# =====================================================
try:
    logger.info("\n📊 Evaluating model on test set...\n")
    eval_results = model.evaluate(test_data, verbose=1)
    test_loss, test_acc, test_top3_acc = eval_results[0], eval_results[1], eval_results[2]
    
    logger.info(f"✅ Test Loss: {test_loss:.4f}")
    logger.info(f"✅ Test Accuracy: {test_acc:.4f}")
    logger.info(f"✅ Test Top-3 Accuracy: {test_top3_acc:.4f}")

    preds = model.predict(test_data, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_data.classes

    idx2class = {v: k for k, v in train_data.class_indices.items()}
    target_names = [idx2class[i] for i in range(NUM_CLASSES)]

    logger.info("\n📊 Classification Report")
    class_report = classification_report(y_true, y_pred, target_names=target_names)
    logger.info(f"\n{class_report}")

    conf_matrix = confusion_matrix(y_true, y_pred)
    logger.info("\n📉 Confusion Matrix")
    logger.info(f"\n{conf_matrix}")

    # Save metrics to JSON
    metrics_dict = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_top3_accuracy': float(test_top3_acc),
        'class_report': classification_report(y_true, y_pred, target_names=target_names, output_dict=True),
        'confusion_matrix': conf_matrix.tolist()
    }

    metrics_path = os.path.join(MODEL_DIR, f'metrics_{TIMESTAMP}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"✅ Metrics saved to {metrics_path}")
    
except Exception as e:
    logger.error(f"❌ Error during evaluation: {e}")
    sys.exit(1)

# =====================================================
# SAVE FINAL MODEL & TRAINING HISTORY
# =====================================================
try:
    model_path = os.path.join(MODEL_DIR, "footprint_cnn_final.keras")
    model.save(model_path)
    logger.info(f"✅ Model saved to {model_path}")

    # Save training history
    history_dict = {
        'phase1': history_phase1.history if hasattr(history_phase1, 'history') else {},
        'phase2': history_phase2.history if hasattr(history_phase2, 'history') else {}
    }
    history_path = os.path.join(MODEL_DIR, f'training_history_{TIMESTAMP}.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"✅ Training history saved to {history_path}")

    # Save training metadata for reproducibility
    metadata = {
        'timestamp': TIMESTAMP,
        'config': config,
        'num_classes': NUM_CLASSES,
        'class_indices': train_data.class_indices,
        'training_samples': len(train_data.classes),
        'validation_samples': len(val_data.classes),
        'test_samples': len(test_data.classes),
        'final_metrics': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_top3_accuracy': float(test_top3_acc)
        }
    }
    metadata_path = os.path.join(MODEL_DIR, f'training_metadata_{TIMESTAMP}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Training metadata saved to {metadata_path}")

except Exception as e:
    logger.error(f"❌ Error saving model and artifacts: {e}")
    sys.exit(1)

# =====================================================
# FINAL SUMMARY
# =====================================================
logger.info("\n" + "="*60)
logger.info("✅ EfficientNetV2-S model training complete!")
logger.info("="*60)
logger.info(f"📁 Checkpoint: {MODEL_DIR}")
logger.info(f"📊 Run TensorBoard: tensorboard --logdir={os.path.join(MODEL_DIR, f'logs_{TIMESTAMP}')}")
logger.info(f"⏱️  Training timestamp: {TIMESTAMP}")
logger.info("="*60)
