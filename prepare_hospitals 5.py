import os
import shutil
import random
import argparse

def prepare_data(input_dir, output_dir, clients):
    classes = ['NORMAL', 'PNEUMONIA']
    
    for client_id in range(1, clients + 1):
        for cls in classes:
            dest_path = os.path.join(output_dir, f"hospital_{client_id}", cls)
            os.makedirs(dest_path, exist_ok=True)
    
    # Gather all images for each class
    image_paths = {cls: [] for cls in classes}
    for cls in classes:
        class_dir = os.path.join(input_dir, cls)
        images = os.listdir(class_dir)
        image_paths[cls] = [os.path.join(class_dir, img) for img in images]
    
    # Distribute images equally among clients
    for cls in classes:
        random.shuffle(image_paths[cls])
        
        for idx, img_path in enumerate(image_paths[cls]):
            client_id = (idx % clients) + 1
            dest_path = os.path.join(output_dir, f"hospital_{client_id}", cls)
            shutil.copy(img_path, dest_path)
    
    print("✅ Data preparation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--clients', type=int, required=True)
    args = parser.parse_args()

    prepare_data(args.input_dir, args.output_dir, args.clients)
