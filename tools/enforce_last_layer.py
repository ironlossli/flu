import glob
import re
import yaml

def get_layer_count(content, filename):
    # Try to parse yaml to find layer count
    try:
        data = yaml.safe_load(content)
    except:
        return None

    # Mapping of backbone specific layer keys
    if 'egnn' in filename:
        return data.get('layers', 6)
    elif 'schnet' in filename:
        return data.get('schnet_layers', 6)
    elif 'painn' in filename:
        # PaiNN config might store it in adapter_kwargs or top level depending on how it was cleaned
        # Let's check typical keys
        return data.get('num_interactions', 6) 
    elif 'leftnet' in filename:
        return data.get('num_layers', 4)
    elif 'equiformer' in filename:
        return data.get('num_layers', 12) # V2 default
    
    return 6 # Default Fallback

config_files = glob.glob("configs/model/*.yaml")

for filepath in config_files:
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Get layer count
    n_layers = get_layer_count(content, filepath)
    if n_layers is None:
        print(f"Skipping {filepath} (cannot parse layers)")
        continue
        
    print(f"Processing {filepath}: {n_layers} layers")
    
    # Regex to replace film_layers: [...]
    # We replace it with film_layers: [N]
    new_line = f"film_layers: [{n_layers}]"
    
    if "film_layers:" in content:
        content = re.sub(r'film_layers:\s*\[.*?\]', new_line, content)
    else:
        # If key doesn't exist, append it after deep_film
        content = re.sub(r'(deep_film:.*)', f'\1\n{new_line}', content)
        
    with open(filepath, 'w') as f:
        f.write(content)
