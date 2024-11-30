import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
import torch
from enum import Enum
import time
import glob

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = '<OPEN_VOCABULARY_DETECTION>'
    """Detect bounding box for objects and OCR text"""

def run_example(task_prompt: TaskType, image, text_input, model, processor, device):
    """Runs an inference task using the model."""
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'Starting inference task...')
    print(time.strftime('%Y-%m-%d %H:%M:%S'), f"Running example with text input: {text_input}")
    prompt = task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device).to(torch.float32) if k != "input_ids" else v.to(device).to(torch.int64) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt.value,
        image_size=(image.width, image.height)
    )
    print(time.strftime('%Y-%m-%d %H:%M:%S'), f"Completed example with text input: {text_input}")
    return parsed_answer


def get_watermark_mask(image, model, processor, device, text_inputs):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'Starting watermark mask generation...')
    task_prompt = TaskType.OPEN_VOCAB_DETECTION  # Use OPEN_VOCAB_DETECTION
    mask = Image.new("L", image.size, 0)  # "L" mode for single-channel grayscale
    draw = ImageDraw.Draw(mask)

    # Obtener dimensiones de la imagen
    image_width, image_height = image.size
    total_image_area = image_width * image_height

    # Bounding box centrado
    box_width = int(image_width * 0.4)
    box_height = int(image_height * 0.3)
    x_center = image_width // 2
    y_center = image_height // 2
    x1_centered = x_center - box_width // 2
    y1_centered = y_center - box_height // 2
    x2_centered = x_center + box_width // 2
    y2_centered = y_center + box_height // 2

    # Dibujar bounding box centrado
    draw.rectangle([x1_centered, y1_centered, x2_centered, y2_centered], fill=255)

    for text_input in text_inputs:
        print(time.strftime('%Y-%m-%d %H:%M:%S'), f'Start processing text input: {text_input}')
        parsed_answer = run_example(task_prompt, image, text_input, model, processor, device)

        detection_key = '<OPEN_VOCABULARY_DETECTION>'
        if detection_key in parsed_answer and 'bboxes' in parsed_answer[detection_key]:
            print(time.strftime('%Y-%m-%d %H:%M:%S'), f'Found bounding boxes for text input: {text_input}')
            for bbox in parsed_answer[detection_key]['bboxes']:
                x1, y1, x2, y2 = map(int, bbox)  # Convert float bbox to int

                # Calculate the area of the bounding box
                bbox_area = (x2 - x1) * (y2 - y1)

                # If the area of the bounding box is less than 10% of the image area, include it in the mask
                if bbox_area <= 0.3 * total_image_area:
                    draw.rectangle([x1, y1, x2, y2], fill=255)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), f'Drawing bounding box: {bbox}')  # Draw a white rectangle on the mask
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), f"Added bounding box to mask: {bbox}")
                else:
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), f"Skipping region: Bounding box covers more than 10% of the image. BBox Area: {bbox_area}, Image Area: {total_image_area}")
        else:
            print(time.strftime('%Y-%m-%d %H:%M:%S'), f"No bounding boxes found in parsed answer for text input '{text_input}'.")

    print(time.strftime('%Y-%m-%d %H:%M:%S'), "Completed watermark mask generation.")
    return mask


def process_image_with_lama(image, mask, model_manager):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'Starting image processing with LaMa...')
    config = Config(
        ldm_steps=50,  # Increased steps for higher quality
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,  # Use CROP strategy for higher quality
        hd_strategy_crop_margin=64,  # Increase crop margin to provide more context
        hd_strategy_crop_trigger_size=800,  # Higher trigger size for larger images
        hd_strategy_resize_limit=1600,  # Increase limit for processing larger images
    )
    result = model_manager(image, mask, config)

    # Ensure result is in the correct format
    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)

    print(time.strftime('%Y-%m-%d %H:%M:%S'), "Completed processing with LaMa model.")
    return result


def main():
    # Parse command line arguments
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Watermark Remover')
    parser.add_argument('input_directory', type=str, help='Path to input directory containing images')
    parser.add_argument('output_directory', type=str, help='Path to save output images')
    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory

    # Check if input directory exists
    if not os.path.exists(input_directory):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), f"Input directory {input_directory} does not exist.")
        sys.exit(1)

    # Create output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all image files in the input directory
    image_paths = glob.glob(os.path.join(input_directory, "*.*"))

    if len(image_paths) == 0:
        print(time.strftime('%Y-%m-%d %H:%M:%S'), f"No images found in directory {input_directory}.")
        sys.exit(1)

    # Load Florence2 model and processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(time.strftime('%Y-%m-%d %H:%M:%S'), f"Using device: {device}")

    print(time.strftime('%Y-%m-%d %H:%M:%S'), "Loading Florence2 model and processor...")
    florence_model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Florence-2-large', trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    florence_model.eval()
    florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)

    # Load LaMa model
    print(time.strftime('%Y-%m-%d %H:%M:%S'), "Loading LaMa model...")
    model_manager = ModelManager(name="lama", device=device)

    # Process each image in the input directory
    text_inputs = ['logo', 'watermark', 'text']

    for image_path in image_paths:
        print(time.strftime('%Y-%m-%d %H:%M:%S'), f"Processing image: {image_path}")
        
        # Load the image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(time.strftime('%Y-%m-%d %H:%M:%S'), f"Failed to load image {image_path}. Skipping. Error: {e}")
            continue

        # Generate watermark mask
        print(time.strftime('%Y-%m-%d %H:%M:%S'), "Generating watermark mask...")
        mask_image = get_watermark_mask(image, florence_model, florence_processor, device, text_inputs)
        mask_image.save("mask.png")
        # Process image with LaMa
        print(time.strftime('%Y-%m-%d %H:%M:%S'), "Processing image to remove watermarks...")
        result_image = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)

        # Convert the result from BGR to RGB
        print(time.strftime('%Y-%m-%d %H:%M:%S'), "Converting result image from BGR to RGB...")
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # Convert result_image from NumPy array to PIL Image
        result_image_pil = Image.fromarray(result_image_rgb)

        # Save the final processed image
        output_image_path = os.path.join(output_directory, os.path.basename(image_path))
        print(time.strftime('%Y-%m-%d %H:%M:%S'), f"Saving the final processed image to {output_image_path}...")
        result_image_pil.save(output_image_path)

    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print(time.strftime('%Y-%m-%d %H:%M:%S'), f"All images processed. Total processing time: {elapsed_time_minutes:.2f} minutes")


if __name__ == '__main__':
    main()
