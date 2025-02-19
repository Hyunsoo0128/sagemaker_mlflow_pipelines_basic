import argparse
import os
import sys
import subprocess
import datetime
import shutil
import logging
import boto3
import torch

DATETIME_STRING = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

MODEL_DIR = '/opt/ml/model'
LOCAL_WEIGHTS_SAVE_DIR = f"cv_weights/{DATETIME_STRING}"
BEST_MODEL_PATH = os.path.join(LOCAL_WEIGHTS_SAVE_DIR, 'train/weights', 'best.pt')
SAGEMAKER_MODEL_PATH = os.path.join(MODEL_DIR, 'model.pt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_packages():
    """ Install necessary Python packages. This step can be avoided by providing a docker image with ultralytics installed. """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        logging.info("Packages installed successfully.")
    except subprocess.CalledProcessError:
        logging.error("Failed to install packages.")
        sys.exit(1)

def upload_directory_to_s3(local_directory, s3_prefix, BUCKET_NAME):
    """ Uploads a directory to an S3 bucket. """
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_prefix, relative_path)
            try:
                s3_client.upload_file(local_path, BUCKET_NAME, s3_path)
                logging.info(f"Uploaded {filename} to S3 at {s3_path}")
            except Exception as e:
                logging.error(f"Failed to upload {local_path} to S3: {e}")
                return False
    return True
    
def print_gpu_memory():
    if torch.cuda.is_available():
        logging.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f} MB")
        logging.info(f"Allocated GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        logging.info(f"Cached GPU Memory: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        
def train(args):
    try:
        import torch
        torch.cuda.empty_cache()  # GPU 캐시 정리
        print_gpu_memory()

        logging.info("Model training started.")

        from ultralytics import YOLO  # Importing here to ensure packages are installed first
        model = YOLO(args.model)
        model.train(
            data="yolo_code/smoke_config.yaml", 
            epochs=args.epochs, 
            batch=args.batch,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.initial_learning_rate,
            lrf=args.final_learning_rate,
            project=LOCAL_WEIGHTS_SAVE_DIR,
            imgsz=416,  # 이미지 크기 축소
            cache=False,  # 캐시 사용 안 함
            workers=0,    # 워커 수 최소화
            #device='cpu'
        )
        logging.info("Model training completed.")

        if not os.path.exists(BEST_MODEL_PATH):
            logging.error(f"Best model not found at {BEST_MODEL_PATH}")
            return

        BUCKET_NAME = args.bucket_name
        S3_FOLDER_NAME = args.folder_name

        shutil.copyfile(BEST_MODEL_PATH, SAGEMAKER_MODEL_PATH)
        logging.info("Best model copied to SageMaker model directory.")
        upload_directory_to_s3(LOCAL_WEIGHTS_SAVE_DIR, S3_FOLDER_NAME, BUCKET_NAME)
    #except Exception as e:
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error("GPU out of memory error occurred")
        else:
            logging.error(f"Error during training: {e}")
        sys.exit(1)

def main():
    """ Main function to handle workflow logic. """
    install_packages()

    parser = argparse.ArgumentParser(description="Train a YOLO model for smoke detection")
    parser.add_argument('--model', type=str, default="yolov8n.yaml")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='auto')
    parser.add_argument('--initial_learning_rate', type=float, default=0.01)
    parser.add_argument('--final_learning_rate', type=float, default=0.01)
    parser.add_argument('--bucket_name', type=str, required=True)
    parser.add_argument('--folder_name', type=str, required=True)

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()