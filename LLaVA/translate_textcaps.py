# Load TextCaps captions and translate half of them
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import os
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ============================================================================
# Step 1: Load TextCaps data
# ============================================================================
train_file_path = f"{os.environ['PROJECT_ROOT']}/LLaVA/textcaps/TextCaps_0.1_train.json"
val_file_path = f"{os.environ['PROJECT_ROOT']}/LLaVA/textcaps/TextCaps_0.1_val.json"

def translate_half(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Successfully loaded JSON data with {len(data['data'])} items")

    # ============================================================================
    # Step 2: Deduplicate by image_id
    # ============================================================================
    print("\nDeduplicating by image_id...")
    id_to_refs = {}

    for item in data['data']:
        image_id = item['image_id']
        refs = tuple(item['reference_strs'])
        
        if image_id not in id_to_refs:
            id_to_refs[image_id] = {'refs': refs, 'item': item}


    data2 = {'data': []}
    for image_id, info in id_to_refs.items():
        data2['data'].append(info['item'])

    print(f"Original data length: {len(data['data'])}")
    print(f"Deduplicated data length: {len(data2['data'])}")

    # ============================================================================
    # Step 3: Load translation model
    # ============================================================================
    print("\nLoading translation model...")
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def translate(en_sentence):
        """Translate sentence using T5-small model"""
        sentence = "translate English to French: " + str(en_sentence)
        inputs = tokenizer(sentence, return_tensors="pt")
        output = model.generate(**inputs,max_length=100)
        translated = tokenizer.decode(output[0], skip_special_tokens=True)
        return translated

    # ============================================================================
    # Step 4: Preprocess - Expand dataset to one item per reference
    # ============================================================================
    print("\nPreprocessing: Expanding to one item per reference...")
    expanded_data = []

    for item in data2['data']:
        image_id = item['image_id']
        references = item['reference_strs']
        
        # Create one item per reference
        for reference in references:
            new_item = {
                "id": str(image_id),
                "image": f"{image_id}.jpg",
                "reference": reference
            }
            expanded_data.append(new_item)

    print(f"Expanded from {len(data2['data'])} items to {len(expanded_data)} items (one per reference)")

    # ============================================================================
    # Step 5: Translate half of the captions
    # ============================================================================
    print("\nTranslating half of the captions...")
    total_items = len(expanded_data)
    half_point = total_items // 2

    formatted_data = []

    for i, item in enumerate(expanded_data):
        image_id = item['id']
        reference = item['reference']
        
        # Translate if in first half, otherwise keep original
        if i < half_point:
            translated_caption = translate(reference)
        else:
            translated_caption = reference
        
        formatted_item = {
            "id": image_id,
            "image": item['image'],
            "conversations": [
                {
                    "from": "human", 
                    "value": "<image>\nPlease carefully observe the image and come up with a caption for the image."
                },
                {
                    "from": "gpt",
                    "value": translated_caption
                }
            ]
        }
        formatted_data.append(formatted_item)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_items} items...")

    print(f"\nCompleted! Translated {half_point} out of {total_items} captions.")
    print(f"Total formatted items: {len(formatted_data)}")

    random.shuffle(formatted_data['data'])
    return formatted_data

train_half_trans=translate_half(train_file_path)
val_half_trans=translate_half(val_file_path)

with open('LLaVA/playground/data/textCapsTrainTranslatedHalf.json', 'w') as f:
    json.dump(train_half_trans['data'], f)
    
with open('LLaVA/playground/data/textCapsValTranslatedHalf.json', 'w') as f:
    json.dump(val_half_trans['data'], f)