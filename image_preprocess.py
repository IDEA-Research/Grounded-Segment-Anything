import os, sys, shutil

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO


import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,BitsAndBytesConfig
import openai as OpenAI
# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import re

from huggingface_hub import hf_hub_download

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import extract_number,get_sorted_files,numpy_to_base64, add_grid_to_image, encode_image


DEVICE_0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DEVICE_1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def count_objects(sentence):
    # Use regular expression to find all occurrences of the pattern ending with "."
    matches = re.findall(r'[^.]+\s*\.', sentence)
    return len(matches)


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model 




# Load Grounding DINO
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, DEVICE_0)


# Load SAM
sam_checkpoint = './sam_weight.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(DEVICE_0))



with open("../GPT-API-Key.txt", "r") as f:
    api_key = f.read().strip()

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def get_result_from_VLM(model,tokenizer,messages):

    inputs = tokenizer.apply_chat_template(messages,
                                    add_generation_prompt=True, 
                                    tokenize=True, 
                                    return_tensors="pt",
                                    return_dict=True)
    inputs = inputs.to(DEVICE_0)
    gen_kwargs = {"max_length": 2500, "do_sample": False, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response=tokenizer.decode(outputs[0])

    


    return response[:-13]


def communicate_gpt(messages,headers=headers):
    """
    get response from GPT-4 API. Content is the input to the API.
    
    """
    
    payload={
            "model":"gpt-4o",
            "messages":messages,
            "max_tokens": 1024,
            "temperature":0,
            "top_p":1,
            "frequency_penalty":0,
            "presence_penalty":0
        }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    
    if 'choices' in response.json():
        # GPT doesn't run into error.                
        response_message=response.json()['choices'][0]['message']['content']
    else:
        print(response.json())
        response_message=None   
        
    return response_message


def VLM_guided_detect(image_tensor, image_np, text_prompt, VLM_model, VLM_tokenizer,cv_model=groundingdino_model, box_threshold = 0.2, text_threshold = 0.2):

    system_prompt="""
        Role: You will provide some guidance on which object to segment for vision models.
        Task: You will be given a natural language description about the task and an image.Now based on the description and the image, you will provide the objects that you think are important to completing the task.
        Output Requirement: Only output the objects description that you think are important to completing the task. For each object, separate them with a space and a period.
        
        I am giving you an example now.
        
        Example 1:
        Task Description: pick up the yellow cup.
        Output: yellow cup . 
        
        
        Example 2:
        Task Description: Move the silver vessel and place it below the spoon.
        Output: silver vessel . spoon .
        

        
        
    """
    
    
    
    another_example="""        
        Example 3:
        Task Description: Build a tool hang by first picking up the L-shaped pole and then piercing it through the hole in the wooden stand, then hang the tool to the tip of the L-shaped pole.
        Output: L-shaped pole. hole. wooden stand. tool.
        """
        
    user_prompt=f"""
    Now, with the given task description and the image, please provide the objects that you think are important to completing the task.
    
    Task Description: {text_prompt}.
    """
    
    
    if VLM_model is None:
        print("using GPT-4o now.")
        
        messages=[{
            "role":"system",
            "content":system_prompt,
        }]
        messages.append({
        "role":"user",
        "content":[user_prompt,{"image":numpy_to_base64(image_np)}],
        })
        
        response=communicate_gpt(messages,headers=headers)
        
    else:   
        messages=[{
            "role":"system",
            "content":system_prompt,
            # "image":Image.open("./VLM_example_image.jpg")
        }]
        

        messages.append({
            "role":"user",
            "content":user_prompt,
            "image":Image.fromarray(image_np)
        })
        response= get_result_from_VLM(VLM_model,VLM_tokenizer,messages)
        
        
    print(f"previous description: {text_prompt}")
    print(f"trimmed down version:{response}")
    annotated_frames, boxes=detect(image_tensor, image_np, response, cv_model, box_threshold, text_threshold)
    
    return annotated_frames,boxes,response


def detect(image_tensor, image_np, text_prompt, model, box_threshold = 0.2, text_threshold = 0.2):
  boxes, logits, phrases = predict(
      model=model, 
      image=image_tensor, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )

  annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
  return annotated_frame, boxes 


def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(DEVICE_0), image.shape[:2])
  masks, A, B = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  
  return masks.cpu()
  
##########Reference Functions from GroundedSAM##########
def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def draw_mask_contour(mask, image, random_color=True):
    # Convert mask to a binary image
    image = image.astype(np.uint8)
    mask = mask.numpy()
    # _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    binary_mask = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Decide on the color
    if random_color:
        color = np.random.random(3)
    else:
        color = [30, 144, 255]  # Deep sky blue in BGR format
    # Draw the contours
    cv2.drawContours(image, contours, -1, color, thickness=2)
    return np.array(image), contours

def sample_points_from_contour(contours, num_points=10, use_largest_contour=False, interval=True):
    """
    Sample points from the contour of a mask.

    :param mask: The binary mask from which to extract contours.
    :param num_points: The number of points to sample from the contour.
    :param use_largest_contour: Whether to sample from only the largest contour.
    :return: A list of sampled points (x, y) from the contour.
    """ 
    if not contours:
        return []  # Return an empty list if no contours found

    if use_largest_contour:
        # Find the largest contour based on area
        contours = [max(contours, key=cv2.contourArea)]

    sampled_points = []
    
    
    for contour in contours:
        
        if interval:
            # Calculate the interval for sampling
            interval = len(contour) // num_points
            # Sample points from the contour
            for i in range(0, len(contour), interval):
                point = contour[i][0]  # Contour points are stored as [[x, y]]
                sampled_points.append((point[0], point[1]))
                
                if len(sampled_points) >= num_points:
                    break  # Stop if we have collected enough points
                
        else:
            sampled_points = farthest_point_sampling(contour, num_points)

    return sampled_points

def draw_points(image, points, color=(0, 255, 0), thickness=1):
    # Ensure the image is in the correct format
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Ensure the image has 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw each point
    for point in points:
        cv2.circle(image, point, radius=2, color=color, thickness=-1)

    return image
##########Done##########


def farthest_point_sampling(contour, num_points=5):
    sampled_points = []
    sampled_points.append(contour[0][0])  # Start with the first point
    for _ in range(1, num_points):
        max_dist = -1
        next_point = None
        for point in contour:
            point = point[0]
            min_dist = np.min([np.linalg.norm(point - np.array(sp)) for sp in sampled_points])
            if min_dist > max_dist:
                max_dist = min_dist
                next_point = point
        sampled_points.append(next_point)
    return sampled_points


def sample_points_from_mask(mask, image, num_points=10, use_largest_contour=True, random_color=False, interval=False):
    # Convert mask to a binary image
    image = image.astype(np.uint8)
    # Ensure the image has 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    
    sampled_points_all = []
    centroids = []
    
    fixed_colors=[[30, 144, 255],[255, 0, 0],[0, 255, 0],[0, 0, 255],[255, 255, 0],[0, 255, 255],[255, 0, 255],[255, 255, 255],[0, 0, 0]]
    
    for i in range(mask.shape[0]):
        if random_color:
            color = np.random.random(3) * 255
        else:
            color = fixed_colors[i]
        sub_mask = mask[i][0].numpy()
        binary_mask = sub_mask.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []  # Return an empty list if no contours found

        if use_largest_contour:
            # Find the largest contour based on area
            contours = [max(contours, key=cv2.contourArea)]


        
        for contour in contours:
            sampled_points = []
            M = cv2.moments(contour)
            
            # Check for moment area to be zero to avoid division by zero
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
                sampled_points.append((cX, cY))
            else:
                centroids.append((0, 0))

                
            
            if interval:
                # Calculate the interval for sampling
                interval = len(contour) // num_points
                # Sample points from the contour
                for i in range(0, len(contour), interval):
                    point = contour[i][0]  # Contour points are stored as [[x, y]]
                    sampled_points.append((point[0], point[1]))
                    
                    if len(sampled_points) >= num_points:
                        break  # Stop if we have collected enough points
                    
            else:
                sampled_points += farthest_point_sampling(contour, num_points)
            sampled_points_all.append(sampled_points)
            # Draw each point
            for idx, point in enumerate(sampled_points):
                height, width = image.shape[:2]
                cv2.circle(image, point, radius=min(height,width)//100, color=color, thickness=-1)
                text_x = min(point[0] + 5, width - 1)
                text_y = min(point[1] + 5, height - 1)
                text_position = (text_x, text_y)
                cv2.putText(image, f'P{i}_{idx}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,0,0], 1)
    
    
    return sampled_points_all,image

   



def segmentation_from_text(image_tensor,image_np, text_prompt, grounding_dino_model=groundingdino_model, sam_model=sam_predictor, box_threshold = 0.2, text_threshold = 0.2, VLM_guided=True, VLM_model=None, tokenizer=None):
    
    
    if VLM_guided:
        
        annotated_frame, boxes, response = VLM_guided_detect(image_tensor,image_np, text_prompt, VLM_model,tokenizer,grounding_dino_model, box_threshold, text_threshold)
        num_objects=count_objects(response)
        
        
    else:
        annotated_frame, boxes = detect(image_tensor,image_np, text_prompt, grounding_dino_model, box_threshold, text_threshold)
        
        response=None
            
    if boxes.shape==(0,4):
        print("No object detected")
        return Image.fromarray(image_np), Image.fromarray(image_np), response,0
    
    
    segmented_frame_masks = segment(image=image_np, sam_model=sam_model, boxes=boxes)
    # print(segmented_frame_masks.shape)
    if VLM_guided and segmented_frame_masks.shape[0]<num_objects:
        print("Some objects not detected")
        detection_ratio=segmented_frame_masks.shape[0]/num_objects
    elif VLM_guided and segmented_frame_masks.shape[0]>num_objects:
        print("Extra objects detected")
        detection_ratio=num_objects/segmented_frame_masks.shape[0]
        
    elif VLM_guided and segmented_frame_masks.shape[0]==num_objects:
        print("All objects detected")
        detection_ratio=1
        
    else:
        detection_ratio=1
    
    
    sampled_points_all,annotated_frame_points = sample_points_from_mask(segmented_frame_masks, image_np, num_points=5, use_largest_contour=True, random_color=True, interval=False)
    #   annotated_frame_with_mask, contours = draw_mask_contour(segmented_frame_masks[0][0], annotated_frame,)
    #   points=sample_points_from_contour(contours, num_points=10, use_largest_contour=True, interval=False)
    
    #   annotated_frame_with_points = draw_points(annotated_frame_with_mask, points)
    
    
    
    #   annotated_frame_with_mask = sample_points_from_contour(segmented_frame_masks[0][0], num_points=10, use_largest_contour=True, interval=False)
    return Image.fromarray(annotated_frame), Image.fromarray(annotated_frame_points), response, detection_ratio



# The check_processed_images function can be updated
def check_processed_images(text_description, image):
    system_prompt="""
    Role: You are now being the supervisor to check if all the objects are detected correctly from a VLM. You will be given the objects that the VLM should detect, and the image with the objects detected by the VLM with a box annotator around each detected object. You will need to check if all the objects are detected correctly.
    
    Input Explanation:
    1. Object list: The list of objects that the VLM should detect. Each object is separated by a space and a period. For example, "yellow cup . spoon ." means the VLM should detect a yellow cup and a spoon.
    
    2. Image: The image with the objects detected by the VLM with a box annotator around each detected object.
    
    
    Output Requirement: 
    You will need to check if all the objects are detected correctly. If all the objects are detected correctly, you will need to provide a "Yes" response. If not, you will need to provide a "No" response. Make sure only to provide a "Yes" or "No" response and nothing else. I am using a hard-coded pattern to match your response so please output exactly as requested.
        
    Example output 1 (all objects detected correctly):
    Detect Result: Yes
    
    
    Example output 2 (not all objects detected correctly):
    Detect Result: No
    
    
    """
    
    user_prompt=f"""Here are the objects to detect:
    
    Objects: {text_description}.
    
    And here's the image with the objects detected by the VLM
    
    """
    
    
    messages=[{
        "role":"system",
        "content":system_prompt,
        
    }]
    
    messages.append({
        "role":"user",
        "content":[user_prompt,{"image":numpy_to_base64(image)}],
    })
    
    response=communicate_gpt(messages)
    
    
    # Define the pattern to match "Detect Result: " followed by "Yes" or "No"
    true_pattern = r"Detect Result: Yes"
    false_pattern = r"Detect Result: No"


    matches_yes=re.findall(true_pattern,response)
    matches_no=re.findall(false_pattern,response)
    
    if len(matches_yes)>0:
        return True
    else:
        return False
    
    

def add_grid_to_processed(dataset_folder, grid_size=5):
    sorted_dir=get_sorted_files(dataset_folder,folders=True)
    
    for dir in sorted_dir:
        cur_dir=os.path.join(dataset_folder,dir)
        processed_image_path=os.path.join(cur_dir,"key_points_GPT_guided.png")
        if not os.path.exists(processed_image_path):
            processed_image_path=os.path.join(cur_dir,"key_points_VLM_guided.png")
            
        
        
        processed_image=Image.open(processed_image_path)
        processed_image_with_grid=add_grid_to_image(np.array(processed_image), grid_size=grid_size,add_caption=True)
        processed_image_with_grid=Image.fromarray(processed_image_with_grid)
        processed_image_with_grid.save(os.path.join(cur_dir,"key_points_with_grid.png"))
    

if __name__ == "__main__":
    

    
    def test_on_dataset(dataset_path="../datasets/jaco_play",save_processed_image=True,dataset_name="jaco_play", VLM_guided=False, VLM_model=None,tokenizer=None):
        sorted_dir=get_sorted_files(dataset_path,folders=True)
        detect_successful=0
        detect_ratio=0
        cases_count=len(sorted_dir)
        
        passed_check_demo_dir=os.path.join(dataset_path,"passed_demos")
        os.makedirs(passed_check_demo_dir,exist_ok=True)
        
        for dir in sorted_dir:
            
            cur_dir=os.path.join(dataset_path,dir)
            image_path=os.path.join(cur_dir,"frames","frame_0.png")
            with open (os.path.join(cur_dir,f"{dataset_name}.txt"),"r") as f:
                text_prompt=f.read()
            image_np,image_tensor = load_image(image_path)
            annotated_frame,annotated_frame_points,response,detection_ratio = segmentation_from_text(image_tensor,image_np, text_prompt, box_threshold=0.3, text_threshold=0.25, VLM_guided=VLM_guided,VLM_model=VLM_model,tokenizer=tokenizer)
        
            
            
            if VLM_guided and VLM_model is not None:
                annotated_frame.save(os.path.join(cur_dir,"processed_image_VLM_guided.png"))
                annotated_frame_points.save(os.path.join(cur_dir,"key_points_VLM_guided.png"))
                with open (os.path.join(cur_dir,f"VLM_response.txt"),"w") as f:
                    f.write(response)
            elif VLM_guided and VLM_model is None:
                annotated_frame.save(os.path.join(cur_dir,"processed_image_GPT_guided.png"))
                annotated_frame_points.save(os.path.join(cur_dir,"key_points_GPT_guided.png"))
                with open (os.path.join(cur_dir,f"GPT_response.txt"),"w") as f:
                    f.write(response)
            else:
                annotated_frame.save(os.path.join(cur_dir,"processed_image_test.png"))
                annotated_frame_points.save(os.path.join(cur_dir,"key_points.png"))
                
                
            if detection_ratio==1:
                
                check_result=check_processed_images(response, np.array(annotated_frame))
                
                if check_result:
                    detect_successful+=1
                    destination_dir = os.path.join(passed_check_demo_dir, dir)
                    shutil.copytree(cur_dir, destination_dir)
                
            diff_ratio=np.abs(1-detection_ratio)
            detect_ratio+=(1-diff_ratio)
                

        
        print(f"Done. {detect_successful} out of {cases_count} cases detected successfully. Average detection ratio: {detect_ratio/cases_count}")
        print("===========================================")
        print(f"Begin adding the grid to the processed images")
        
        add_grid_to_processed(passed_check_demo_dir, grid_size=5)
        
        
        # sorted_dir_passed=get_sorted_files(passed_check_demo_dir,folders=True)
        
        # for dir in sorted_dir_passed:
            
        
        
       
    parser=argparse.ArgumentParser() 
    # parser.add_argument("--dataset_path", type=str, default="../datasets/jaco_play", help="Path to the dataset")
    parser.add_argument("--save_processed_image", type=bool, default=True, help="Save the processed image")
    parser.add_argument("--dataset_name", type=str, default="jaco_play", help="Name of the dataset")
    parser.add_argument("--VLM_guided", type=int, default=0, help="Use VLM guided detection")
    parser.add_argument("--VLM_model", type=str, default="GPT")
    args=parser.parse_args()
    
    dataset_path=f"../datasets/{args.dataset_name}"
    VLM_guided=bool(args.VLM_guided)
    
    if args.VLM_model=="GPT":
        VLM_model=None
        tokenizer=None
    else:
        # Load VLM
        GLM_path="THUDM/glm-4v-9b"
        tokenizer=AutoTokenizer.from_pretrained(GLM_path, trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        try:
            glm_4v=AutoModelForCausalLM.from_pretrained(GLM_path, quantization_config=quantization_config,device_map=DEVICE_0, trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).eval()
            model=glm_4v
        except torch.cuda.OutOfMemoryError:
            model=None

        VLM_model=model
        
        
    test_on_dataset(dataset_path=dataset_path,save_processed_image=True,dataset_name=args.dataset_name, VLM_guided=VLM_guided,VLM_model=VLM_model,tokenizer=tokenizer)