import json
import os
import glob

def get_key_val(part):
    key_index_start = part.index('[')
    key_index_end = part.index(']')
    val_index_start = part.rindex('(')
    val_index_end = part.rindex(')')

    key = part[key_index_start + 1: key_index_end]
    val = part[val_index_start + 1: val_index_end]

    return key, val


def add_pytorch_tensorflow(model):
    # handle environment part
    env_part = model["Environment"].split(', ')[0]
    
    search_dict = {
            "PyTorch": "torch",
            "TensorFlow": "tensorflow"
    }
    
    if "requirements" in env_part:
        _, requirements_dir = get_key_val(env_part)

        # read requirements file
        with open(f'../{requirements_dir}', 'r') as file:
            requirements = file.read()
            
            for header, package in search_dict.items():
                model[header] = package in requirements
    else:
        for header, _ in search_dict.items():
            model[header] = False


def add_modalities_for_model(model):
    modalities_keywords = {
        "User Text": "user_text",
        "User Image": "user_image",
        "User Graph": "user_graph",
        "Item Text": "item_text",
        "Item Image": "item_image",
        "Item Graph": "item_graph",
        "Sentiment": "sentiment",
        "Review Text": "review_text",
    }
    
    for filename in glob.glob(f'../{model["Link"]}/*.py', recursive=True):
        with open(filename, 'r') as file:
            file_data = file.read()
            
            for header, modality_keyword in modalities_keywords.items():
                is_found = modality_keyword in file_data
                if is_found:
                    model[header] = True
            
            # for user feature and item feature
            # >> if user feature is found, we set user text, image and graph to true
            is_found = "user_feature" in file_data
            if is_found:
                model["User Text"] = True
                model["User Image"] = True
                model["User Graph"] = True
                
            # likewise for item feature
            is_found = "item_feature" in file_data
            if is_found:
                model["Item Text"] = True
                model["Item Image"] = True
                model["Item Graph"] = True
    
    for header, modality_keyword in modalities_keywords.items():
        if header not in model:
            model[header] = False


if __name__ == "__main__":
    # Read the content from README.md
    with open('../README.md', 'r') as file:
        content = file.read()

    # Extract the relevant information from the content
    models = []
    lines = content.split('\n')
    lines = lines[lines.index('## Models') + 4: lines.index('## Resources') - 2]

    headers = []
    headers = lines[0].split('|')[1:-1]
    headers = [header.strip() for header in headers]

    for line in lines[2:]:
        parts = line.split('|')[1:-1]
        parts = [part.strip() for part in parts]
        model = dict(zip(headers, parts))
        models.append(model)

    year = None

    for model in models:
        # handle empty years
        if model["Year"] == "":
            model["Year"] = year
        else:
            year = model["Year"]
            
        # handle model, docs and paper part
        name_paper_str = model["Model and Paper"]

        for i, part in enumerate(name_paper_str.split(', ')):
            key, val = get_key_val(part)

            if i == 0:
                model["Name"] = key
                model["Link"] = val
            else:
                model[key] = val
        
        # Check for PyTorch and TensorFlow in each requirements file
        add_pytorch_tensorflow(model)
        
        # Check for modalities keywords in files and add to model
        add_modalities_for_model(model)
                    
        # remove non required keys
        model.pop("Model and Paper")
        model.pop("Environment")
        
        # Get package name
        model_dir = model["Link"]
        
        with open(f'../{model_dir}/__init__.py', 'r') as file:
            init_data = file.read()
            
            package_names = []
            
            for row in init_data.split('\n'):
                if "import" in row:
                    package_name = row[row.index("import") + len("import "):]
                    package_names.append(f"cornac.models.{package_name}")
            
            model["packages"] = package_names

    json_str = json.dumps(models, indent=4)

    # Write the JSON object to a file
    with open('source/_static/models/data.js', 'w') as file:
        file.write(f"var data = {json_str};")
